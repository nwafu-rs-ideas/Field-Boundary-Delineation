# SAM微调编、解码器测试_层选择_掩膜+解冻、DropPath.py
import os
import glob
import time
import gc
import cv2
from tqdm import tqdm
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from skimage.measure import label as label_cc
import pandas as pd
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation, disk
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch, gc
# SAM (assume segment-anything repo or pip package available)
import sys

sys.path.insert(0, r"D:\AResearchDirection1\segment-anything-main")

# --- SAM 包加载 ---
import importlib
import segment_anything1
import importlib.util

# ✅ 直接导入模块对象，而不是属性引用
build_sam_module = importlib.import_module("segment_anything1.build_sam")
# ✅ 强制重新加载磁盘版本（确保修改过的 build_sam.py 生效）
importlib.reload(build_sam_module)

print("[DEBUG] build_sam loaded from:", build_sam_module.__file__)

# 其他导入
import segment_anything1.modeling.image_encoder as md

print("[DEBUG] image_encoder loaded from:", md.__file__)
# 顶部增加：
from segment_anything1.modeling.image_encoder import TransformerAdapter as EncTransformerAdapter


def run_sam_with_adapter(prompter_model, adapter_dim, LR, EPOCHS, subset_ratio, a, train_head, enc_strategy, do_test):
    # 彻底释放 GPU 占用
    gc.collect()
    torch.cuda.empty_cache()
    # -----------------------
    # Config - EDIT PATHS / PARAMS HERE
    # -----------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAM_CKPT = r"D:\AResearchDirection1\segment-anything-main\sam_vit_h_4b8939.pth"  # 你的 SAM 权重路径（或 vit_h）
    PROMPTER_CKPT = r"D:\AResearchDirection\AI4B\256\8\U-net\test_amplification-masks_0_boundary20319\best5\RG+NIR_newdata2222\checkpoints_" + f"{prompter_model}" + "\prompter_" + f"{prompter_model}" + "_best.pth"

    # DeepLabV3 r"D:\AResearchDirection\AI4B\AAAAA\DeepLabV3\test_amplification-masks_0_boundary20319\best5\RG+NIR\checkpoints_" + f"{prompter_model}" + "\prompter_" + f"{prompter_model}" + "_best.pth
    # 筛后数据集 r"D:\AResearchDirection\AI4B\256\8\U-net\test_amplification-masks_0_boundary20319\best5\RG+NIR_newdata2222\checkpoints_" + f"{prompter_model}" + "\prompter_" + f"{prompter_model}" + "_best.pth"
    # 3波段b1 D:\AResearchDirection\AI4B\256\8\U-net\dataset_add\test_amplification-dataset_add_masks_0
    # 3波段b2 D:\AResearchDirection\AI4B\256\8\U-net\test_amplification-masks_0_boundary2\
    # fr"D:/AResearchDirection/AI4B/256/8/U-net/test_{a}-masks_0/checkpoints_U-net/unet_best.pth"
    # 4波段b1 D:\AResearchDirection\AI4B\256\8\U-net\4_inchans\test_amplification-masks_0_boundary2\
    # 数据路径（train/val/test）
    def pair_by_stem(nc_glob, region_glob, boundary_glob):
        """
        按“主干名”对齐：只保留同时存在 nc / region / boundary 的样本。
        region 允许带后缀 '_region'，boundary 允许带后缀 '_boundary'。
        返回 (paired_nc, paired_rg, paired_bd) 已按主干名排序且一一对应。
        """

        def stem_nc(p):
            return os.path.splitext(os.path.basename(p))[0]

        def stem_region(p):
            # 例：xxx_region.tif -> xxx
            s = os.path.splitext(os.path.basename(p))[0]
            return s

        def stem_boundary(p):
            # 例：xxx_boundary.tif -> xxx
            s = os.path.splitext(os.path.basename(p))[0]
            return s.replace("_boundary", "")

        nc_list = sorted(glob.glob(nc_glob))
        rg_list = sorted(glob.glob(region_glob))
        bd_list = sorted(glob.glob(boundary_glob))

        nc_map = {stem_nc(p): p for p in nc_list}
        rg_map = {stem_region(p): p for p in rg_list}
        bd_map = {stem_boundary(p): p for p in bd_list}

        keys = sorted(set(nc_map) & set(rg_map) & set(bd_map))

        # 打印缺失情况（可选）
        miss_nc = sorted((set(rg_map) | set(bd_map)) - set(nc_map))
        miss_rg = sorted(set(nc_map) - set(rg_map))
        miss_bd = sorted(set(nc_map) - set(bd_map))
        if miss_nc:
            print(f"⚠ 有 {len(miss_nc)} 个样本只有标注/边界没有影像（忽略）：", miss_nc[:5], "...")
        if miss_rg:
            print(f"⚠ 有 {len(miss_rg)} 个样本只有影像没有 region（忽略）：", miss_rg[:5], "...")
        if miss_bd:
            print(f"⚠ 有 {len(miss_bd)} 个样本只有影像没有 boundary（忽略）：", miss_bd[:5], "...")

        paired_nc = [nc_map[k] for k in keys]
        paired_rg = [rg_map[k] for k in keys]
        paired_bd = [bd_map[k] for k in keys]

        print(f"✅ 成功按文件名对齐样本数：{len(keys)}（将严格一一对应使用）")
        return paired_nc, paired_rg, paired_bd

    # 按文件名对齐，仅保留真正一一对应的样本
    # 按文件名对齐，仅保留真正一一对应的样本
    TRAIN_NC, TRAIN_REGION, TRAIN_BOUNDARY = pair_by_stem(  # dataset_add\
        r"D:\AResearchDirection\AI4B\train_filter\*.nc",
        r"D:\AResearchDirection\AI4B\dataset_0506\masks_region\train\*.tif",
        r"D:\AResearchDirection\AI4B\dataset_0506\masks_boundary2222\train\*.tif",
    )

    VAL_NC, VAL_REGION, VAL_BOUNDARY = pair_by_stem(
        r"D:\AResearchDirection\AI4B\val_filter\*.nc",
        r"D:\AResearchDirection\AI4B\dataset_0506\masks_region\val\*.tif",
        r"D:\AResearchDirection\AI4B\dataset_0506\masks_boundary2222\val\*.tif",
    )

    TEST_NC_DIR = r"D:\AResearchDirection\AI4B\test_filter"
    TEST_MASK_DIR = r"D:\AResearchDirection\AI4B\dataset_0506\masks_region\test"
    TEST_BOUNDARY = r"D:\AResearchDirection\AI4B\dataset_0506\masks_boundary2222\test"

    OUT_DIR = fr"D:\AResearchDirection\AI4B\AAAAA\{train_head}wu1_yanmoprompt_adapter128\newdata_RGNIR\test{enc_strategy}masks_0\sam_adapter_patch_encoder{a}\results_sam_adapter_{prompter_model}"
    CKPT_DIR = fr"D:\AResearchDirection\AI4B\AAAAA\{train_head}wu1_yanmoprompt_adapter128\newdata_RGNIR\test{enc_strategy}masks_0\sam_adapter_patch_encoder{a}\checkpoints_adapter_{prompter_model}\sam_adapter"
    #D:\AResearchDirection\AI4B\256\8\dataset\bothwu1_yanmoprompt_adapter128\unet_boundary2\prompter_prob_mask\jiedong_droppath_boundary2\testsandwich32masks_0\sam_adapter_patch_encoder0.4
    ckpt_path = os.path.join(CKPT_DIR, f"sam_adapter_ckpt_{prompter_model}.pth")
    best_path = os.path.join(CKPT_DIR, f"sam_adapter_best_{prompter_model}.pth")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "tif"), exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Training hyperparams

    WEIGHT_DECAY = 5e-3
    BATCH_SIZE = 2  # effective batch
    ACCUM_STEPS = 4
    accum_steps = max(1, int(os.getenv("ACCUM_STEPS", str(ACCUM_STEPS))))

    WARMUP_STEPS = 256  # linear warmup

    patience = 10  # ⭐ 早停耐心值
    # Misc
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # 全局调试开关
    DEBUG = False  # 调试时设为 True，正常跑实验时设为 False
    DEBUG1 = True

    def debug_print(*args, **kwargs):
        if DEBUG:
            # 如果调用者传入了 flush 参数，使用它；否则默认 True
            if 'flush' not in kwargs:
                kwargs['flush'] = True
            print(*args, **kwargs)

    def debug1_print(*args, **kwargs):
        if DEBUG1:
            print(*args, **kwargs)

    # -----------------------
    # Utility functions (I/O, metrics, tiff save)
    # -----------------------
    def read_rgb_from_nc(nc_path, time_index=0):
        ds = xr.open_dataset(nc_path)
        # 支持 B4/B04, B3/B03, B2/B02
        for bn in ("B4", "B04"):
            if bn in ds:
                b4 = ds[bn][time_index].values;
                break
        for bn in ("B3", "B03"):
            if bn in ds:
                b3 = ds[bn][time_index].values;
                break
        for bn in ("B2", "B02"):
            if bn in ds:
                b2 = ds[bn][time_index].values;
                break
        rgb = np.stack([b4, b3, b2], axis=-1).astype(np.float32)
        ds.close()
        rgb = np.clip(rgb, 0, 3000) / 3000.0
        return rgb

    def read_gt_tif(gt_path):
        with rasterio.open(gt_path) as src:
            gt = src.read(1)
            nodata_val = src.nodata
        if nodata_val is not None:
            mask = (gt != nodata_val)
            gt = np.where(mask, gt, 0)
        return (gt > 0).astype(np.uint8)

    from torch.utils.data.dataloader import default_collate
    def safe_collate1(batch):
        """确保 collate 不会静默跳过坏样本"""
        if any(b is None for b in batch):
            bad_indices = [i for i, b in enumerate(batch) if b is None]
            raise RuntimeError(f"[Collate ERROR] Batch contains invalid samples at indices {bad_indices}")

        # 如果 batch 为空，直接报错
        if len(batch) == 0:
            raise RuntimeError("[Collate ERROR] Empty batch received!")

        return default_collate(batch)

    def compute_pixel_metrics(y_true, y_pred):
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()

        # 如果有效区域为空，返回0指标
        if len(y_true_f) == 0:
            return {"IoU": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0,
                    "TP": 0, "FP": 0, "FN": 0, "TN": 0}
        tp = int(np.logical_and(y_true_f == 1, y_pred_f == 1).sum())
        fp = int(np.logical_and(y_true_f == 0, y_pred_f == 1).sum())
        fn = int(np.logical_and(y_true_f == 1, y_pred_f == 0).sum())
        tn = int(np.logical_and(y_true_f == 0, y_pred_f == 0).sum())
        eps = 1e-7
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        return {"IoU": float(iou), "F1": float(f1), "Precision": float(precision),
                "Recall": float(recall), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


    # ============================================================
    # Boundary 后处理评价指标
    # ============================================================
    BOUNDARY_RADIUS = 4
    BOUNDARY_TOLERANCE = 4

    def boundary_iou_post(pred_b, gt_b):
        """原始 boundary mask 的直接 IoU。保留作为辅助函数。"""
        intersection = np.logical_and(pred_b, gt_b).sum()
        union = np.logical_or(pred_b, gt_b).sum()
        return float(intersection / (union + 1e-8))

    def biou_post(pred_b, gt_b, radius=4):
        pred_b = np.asarray(pred_b).astype(bool)
        gt_b = np.asarray(gt_b).astype(bool)

        pred_band = binary_dilation(pred_b, disk(radius))
        gt_band = binary_dilation(gt_b, disk(radius))

        inter = np.logical_and(pred_band, gt_band).sum()
        union = np.logical_or(pred_band, gt_band).sum()

        return float(inter / (union + 1e-8))

    def boundary_f1_post(pred_b, gt_b, tolerance=4):
        """
        Boundary F-measure with distance tolerance.
        返回：
            f1, precision, recall, TP, FP, FN
        """
        p = np.asarray(pred_b).astype(bool)
        g = np.asarray(gt_b).astype(bool)

        dt_to_gt = distance_transform_edt(~g)
        TP = int(np.sum(p & (dt_to_gt <= tolerance)))
        FP = int(np.sum(p & (dt_to_gt > tolerance)))

        dt_to_pred = distance_transform_edt(~p)
        FN = int(np.sum(g & (dt_to_pred > tolerance)))

        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        return (
            float(f1),
            float(precision),
            float(recall),
            TP,
            FP,
            FN,
        )

    def compute_boundary_post_metrics(pred_b, gt_b,
                                      radius=BOUNDARY_RADIUS,
                                      tolerance=BOUNDARY_TOLERANCE):
        pred_b = np.asarray(pred_b).astype(np.uint8)
        gt_b = np.asarray(gt_b).astype(np.uint8)

        # === 后处理 Boundary IoU ===
        biou3 = biou_post(pred_b, gt_b, radius=radius)

        # === 后处理 Boundary F1 ===
        bf1, bp, br, TP, FP, FN = boundary_f1_post(
            pred_b, gt_b, tolerance=tolerance
        )

        # 只覆盖原 boundary 四个评价值；字段名保持原代码不变。
        metrics = {
            "IoU": biou3,
            "F1": bf1,
            "Precision": bp,
            "Recall": br,
        }

        metrics["TP"] = TP
        metrics["FP"] = FP
        metrics["FN"] = FN
        metrics["TN"] = int(np.logical_and(~pred_b.astype(bool),
                                           ~gt_b.astype(bool)).sum())

        return metrics

    def gt_to_instance(gt_mask):
        return label_cc(gt_mask, connectivity=1)

    def save_label_tif(label, ref_tif_path, out_tif):
        """
        使用标签 tif 的空间参考信息保存预测结果
        （不要再从 nc 的 x/y 推 transform）
        """
        with rasterio.open(ref_tif_path) as ref:
            meta = ref.meta.copy()

        meta.update({
            "dtype": "uint8",
            "count": 1,
            "compress": "lzw",
            "nodata": 0
        })

        with rasterio.open(out_tif, "w", **meta) as dst:
            dst.write(label.astype(np.uint8), 1)

    # EarlyStopping
    class EarlyStopping:
        def __init__(self, patience=patience, delta=0.0, path=best_path):
            self.patience = patience
            self.delta = delta
            self.path = path
            self.counter = 0
            self.best_score = float("inf")  # ⭐ 损失越小越好
            self.early_stop = False

        def __call__(self, val_loss, model):
            if val_loss < self.best_score - self.delta:
                self.best_score = val_loss
                torch.save({"model": model.state_dict()}, self.path)
                print(f"✅ 验证 loss 降低: {val_loss:.6f}, 新的最佳模型已保存到 {self.path}")
                self.counter = 0
            else:
                self.counter += 1
                print(f"⚠️ 验证 loss 未降低: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    print("⏹️ 触发早停")
                    self.early_stop = True

    # -----------------------
    # Loss: Dice + Focal (same as you had)
    # -----------------------
    def sample_points_from_prob(prob, max_points=16):
        device = prob.device
        H, W = prob.shape

        pos_mask = prob > 0.8
        neg_mask = prob <= 0.2

        pos_idx = pos_mask.nonzero(as_tuple=False)
        neg_idx = neg_mask.nonzero(as_tuple=False)

        rng = torch.Generator(device=device)
        rng.manual_seed(int(time.time()) & 0xffff)

        # fallback
        if pos_idx.size(0) == 0:
            pos_idx = torch.stack([
                torch.randint(0, H, (5,), generator=rng, device=device),
                torch.randint(0, W, (5,), generator=rng, device=device)
            ], dim=1)

        if neg_idx.size(0) == 0:
            neg_idx = torch.stack([
                torch.randint(0, H, (5,), generator=rng, device=device),
                torch.randint(0, W, (5,), generator=rng, device=device)
            ], dim=1)

        # 限制数量
        if pos_idx.size(0) > max_points:
            perm = torch.randperm(pos_idx.size(0), generator=rng, device=device)[:max_points]
            pos_idx = pos_idx[perm]

        if neg_idx.size(0) > max_points:
            perm = torch.randperm(neg_idx.size(0), generator=rng, device=device)[:max_points]
            neg_idx = neg_idx[perm]

        pts = torch.cat([pos_idx, neg_idx], dim=0).float()

        labels = torch.cat([
            torch.ones(pos_idx.size(0), dtype=torch.int64, device=device),
            torch.zeros(neg_idx.size(0), dtype=torch.int64, device=device)
        ], dim=0)

        # box
        ys, xs = pos_mask.nonzero(as_tuple=True)
        if ys.numel() > 0:
            box = torch.tensor([
                xs.min(), ys.min(),
                xs.max(), ys.max()
            ], dtype=torch.float32, device=device)
        else:
            box = torch.tensor([0., 0., W - 1., H - 1.], device=device)

        return pts, labels, box
    class RSForSAMDataset(Dataset):
        def __init__(self, nc_paths, region_gt_paths, boundary_gt_paths,
                     prompter_ckpt, prompter_builder,
                     bands=None, crop=256):
            self.nc = []
            self.rg = []
            self.bd = []
            for nc_path, rg_path, bd_path in zip(nc_paths, region_gt_paths, boundary_gt_paths):
                # 打开 region tif 检查是否全 0
                with rasterio.open(rg_path) as src:
                    rg = src.read(1)
                if np.count_nonzero(rg) == 0:
                    # print(f"[FILTER] {rg_path} 全为 0，跳过该样本。")
                    continue
                self.nc.append(nc_path)
                self.rg.append(rg_path)
                self.bd.append(bd_path)

            self.prompter = prompter_builder().to(DEVICE)
            ckpt = torch.load(prompter_ckpt, map_location=DEVICE)
            if "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
            # ===== 修正 key 前缀 =====
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("net."):
                    new_k = "model." + k[4:]  # net.xxx -> model.xxx
                else:
                    new_k = k
                new_state_dict[new_k] = v
            missing, unexpected = self.prompter.load_state_dict(new_state_dict, strict=False)
            print("missing:", missing)
            print("unexpected:", unexpected)
            w = self.prompter.state_dict()['model.encoder.conv1.weight']
            print("weight mean:", w.abs().mean())
            self.prompter.eval()
            self.bands = bands
            self.crop = crop

        def __len__(self):
            return len(self.nc)

        def _load_nc(self, p, time_index=0):
            ds = xr.open_dataset(p)

            def select_band(band_name):
                if band_name not in ds:
                    return None
                data = ds[band_name]
                if "time" in data.dims:
                    return data[time_index].values
                else:
                    return data.values

            b4 = select_band("B4")
            b3 = select_band("B3")
            b2 = select_band("B2")

            b8 = select_band("B8")  # ✅ 新增_inchans
            ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
            ds.close()

            # ===== 2. RG 单独处理 =====
            rg = np.stack([b4, b3], axis=0).astype(np.float32)
            p = np.percentile(rg, 99.8)
            p = max(p, 1e-6)
            rg = np.clip(rg / p, 0, 1.0)
            rg = rg * 255.0

            # --- 单通道 scaling ---
            def scale_band(x):
                x = x.astype(np.float32)
                p = np.percentile(x, 99.8)
                p = max(p, 1e-6)
                x = np.clip(x / p, 0, 1.0)
                return x * 255.0

            b8 = scale_band(b8)
            b8 = b8[np.newaxis, ...]  # ⭐ 关键！
            # ===== 3. NDVI 单独处理 =====
            # ⭐ 强烈建议：拉到 0~255（关键！）
            # ndvi = (ndvi + 1.0) / 2.0  # [-1,1] → [0,1]
            # ndvi = np.clip(ndvi, 0, 1)
            # ndvi = ndvi * 255.0
            # ndvi = ndvi[np.newaxis, ...]  # (1,H,W)   194.7  30.0

            # ===== 4. 拼接 =====
            img = np.concatenate([rg, b8], axis=0).astype(np.float32)  # (3,H,W)
            # ===== 5. 标准化（⚠️要改！）=====
            mean = np.array([123.675, 116.28, 127.95], dtype=np.float32).reshape(3, 1, 1)  # mean RGB（123.675, 116.28, 103.53）RGB、B8 85.311, 95.33, 78.21,127.95
            std = np.array([58.395, 57.12, 47.64], dtype=np.float32).reshape(3, 1, 1)  # std  RG（58.395, 57.12, 57.375）   RGB、B8 47.564, 44.34, 43.37, 47.64
            img = (img - mean) / std
            return img

        @torch.no_grad()
        def _make_prompts(self, img_t):
            """
            输入: img_t Tensor (3,H,W) 或 (1,3,H,W)
            输出: mask_prompt_logits (Tensor 1,2,H,W), pts (Tensor N,2), labels (Tensor N), box (Tensor 4)
            这个实现做了类型检查、fallback、采样上限，避免 KeyError / empty/nonzero 卡死 等问题。
            """
            # 保证 batch 维度
            if img_t.dim() == 3:
                img_in = img_t.unsqueeze(0)
            else:
                img_in = img_t

            device = next(self.prompter.parameters()).device if hasattr(self, 'prompter') else img_in.device

            with torch.no_grad():
                # 调用 prompter：返回类型可能是 Tensor / tuple / dict，根据具体实现调整
                out = self.prompter(img_in)  # <= 可能返回 dict/tuple/tensor

                # 兼容各种返回
                if isinstance(out, dict):
                    if 'out' in out:
                        mask_prompt_logits = out['out']  # ⭐ 主输出
                    elif 'masks' in out:
                        mask_prompt_logits = out['masks']
                    elif 'logits' in out:
                        mask_prompt_logits = out['logits']
                    elif 'pred_mask' in out:
                        mask_prompt_logits = out['pred_mask']
                    else:
                        raise RuntimeError(f"未知dict keys: {list(out.keys())}")
                elif isinstance(out, (list, tuple)):
                    # ⭐ 关键修改
                    assert len(out) == 2, f"期望 (region, boundary)，实际是 {len(out)} 个输出"
                    out_r, out_b = out
                    assert torch.is_tensor(out_r) and torch.is_tensor(out_b), "tuple元素必须是tensor"
                    mask_prompt_logits = torch.cat([out_r, out_b], dim=1)
                elif torch.is_tensor(out):
                    mask_prompt_logits = out
                else:
                    raise RuntimeError(f"未知 prompter 返回类型: {type(out)}")
                assert mask_prompt_logits.shape[1] == 2, \
                    f"期望2通道(region+boundary)，实际是 {mask_prompt_logits.shape}"

                # 确保为 tensor 且带 batch 维度
                if not torch.is_tensor(mask_prompt_logits):
                    mask_prompt_logits = torch.as_tensor(mask_prompt_logits)

                # 期望 shape 是 (B, C, H, W) 或 (C, H, W)
                if mask_prompt_logits.dim() == 3:
                    # (C,H,W) -> (1,C,H,W)
                    mask_prompt_logits = mask_prompt_logits.unsqueeze(0)

                B, C, H, W = mask_prompt_logits.shape
                assert C == 2, f"通道数错误: {C}"

                # 选择 region 通道（如果 C>=1）
                # 这通道 0 是 region，通道 1 是 boundary；
                region_channel_index = 0
                if C <= region_channel_index:
                    raise RuntimeError(f"prompter 输出通道数太小: {C}")

            prob_region = torch.sigmoid(mask_prompt_logits[0, 0]) > 0.5
            prob_boundary = torch.sigmoid(mask_prompt_logits[0, 1]) > 0.5

            # ⭐ 分别生成
            pts_r, labels_r, box_r = sample_points_from_prob(prob_region)
            pts_b, labels_b, box_b = sample_points_from_prob(prob_boundary)

            # 返回：保持 mask_prompt_logits 在 cpu 上或原 device（按你 pipeline 要求）
            return (
                mask_prompt_logits,
                pts_r, labels_r, box_r,
                pts_b, labels_b, box_b
            )

        def __getitem__(self, i):
            try:
                img = self._load_nc(self.nc[i])  # numpy (3,H,W)
                # ---- 读取 region ----
                with rasterio.open(self.rg[i]) as src:
                    rg = src.read(1).astype(np.int64)

                # ---- 读取 boundary ----
                with rasterio.open(self.bd[i]) as src:
                    bd = src.read(1).astype(np.int64)

                if isinstance(img, np.ndarray):
                    img_t = torch.from_numpy(img)
                elif isinstance(img, torch.Tensor):
                    img_t = img
                else:
                    raise TypeError(f"Unexpected type for img: {type(img)}")

                # === 调用 prompter 生成提示 ===
                mask_prompt_logits, pts_r, labels_r, box_r, pts_b, labels_b, box_b = self._make_prompts(
                    img_t.unsqueeze(0).float().to(DEVICE)  # ⭐ 保证是 (1,3,H,W)
                )

                debug_print(f"[DEBUG __getitem__] i={i}, img_t.shape={img_t.shape}, "
                            f"pts.shape={pts_r.shape}, labels.shape={labels_r.shape}, box={box_r}")

                region_prompt = torch.sigmoid(mask_prompt_logits[0:1, 0]).cpu() > 0.8
                boundary_prompt = torch.sigmoid(mask_prompt_logits[0:1, 1]).cpu() > 0.8

                return (
                    img_t,
                    torch.from_numpy(rg),
                    torch.from_numpy(bd),
                    mask_prompt_logits.squeeze(0).cpu(),

                    pts_r.cpu(),
                    labels_r.cpu(),
                    box_r.cpu(),

                    pts_b.cpu(),
                    labels_b.cpu(),
                    box_b.cpu(),

                    region_prompt,
                    boundary_prompt,

                    self.nc[i]
                )

            except Exception as e:
                import traceback
                debug1_print(f"[ERROR] 样本 {i} 出错: {self.nc[i]}, 错误={e}")
                traceback.print_exc()
                raise RuntimeError(f"数据集样本 {i} 加载失败") from e

    class DiceLoss(nn.Module):
        def __init__(self, eps=1e-6):
            super().__init__();
            self.eps = eps

        def forward(self, preds, targets, reduction="mean"):
            preds = preds.view(preds.size(0), -1)
            targets = targets.view(targets.size(0), -1).float()
            inter = (preds * targets).sum(1)
            union = preds.sum(1) + targets.sum(1) + self.eps
            dice = 2 * inter / union
            loss = 1 - dice  # [B]

            if reduction == "none":
                return loss
            elif reduction == "mean":
                return loss.mean()
            else:  # "sum"
                return loss.sum()

    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, logits, targets, reduction='mean'):
            # logits: [B, 1, H, W] or [B, H, W]
            # targets: [B, H, W] (binary)

            targets = targets.float().unsqueeze(1) if logits.dim() == 4 else targets.float()
            # ---- 确保 logits 和 targets 维度匹配 ----
            if logits.dim() == 3:
                logits = logits.unsqueeze(1)  # [B,1,H,W]
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)  # [B,1,H,W]
            targets = targets.float()
            if logits.shape != targets.shape:
                debug1_print(f"Shape mismatch: logits {logits.shape}, targets {targets.shape}")

            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
            pt = torch.exp(-bce)  # [B, 1, H, W]

            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * (1 - pt) ** self.gamma * bce

            # ---- 每个样本的平均 loss ----
            focal = focal.mean(dim=[1, 2, 3])  # → [B]

            if reduction == 'none':
                return focal  # per-pixel loss
            elif reduction == 'mean':
                return focal.mean()
            elif reduction == 'sum':
                return focal.sum()
            else:
                raise ValueError(f"Invalid reduction: {reduction}")

    # -----------------------
    # Build SAM and apply adapter to mask_decoder modules
    # -----------------------
    class TransformerAdapter(nn.Module):
        def __init__(self, hidden_dim=1280, bottleneck_dim=64):
            super().__init__()
            self.down = nn.Linear(hidden_dim, bottleneck_dim)
            self.act = nn.ReLU()
            self.up = nn.Linear(bottleneck_dim, hidden_dim)

        def forward(self, x):
            return x + self.up(self.act(self.down(x)))

    class Adapter(nn.Module):
        def __init__(self, input_dim, adapter_dim, dropout=0.1, nonlinearity="relu"):
            super().__init__()
            self.down = nn.Linear(input_dim, adapter_dim)
            self.activation = nn.ReLU() if nonlinearity == "relu" else nn.GELU()
            self.up = nn.Linear(adapter_dim, input_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.dropout(self.up(self.activation(self.down(x))))

    def add_adapters(module, adapter_dim, dropout=0.1):
        """
        递归替换 nn.Linear -> nn.Sequential(Linear(named 'linear'), Adapter(named 'adapter'))
        这样命名后，adapter 参数名会包含 'adapter'，便于按 name 筛选。
        """
        for name, child in list(module._modules.items()):
            # 递归处理子模块
            if len(list(child.children())) > 0:
                add_adapters(child, adapter_dim, dropout)
            # 如果是 Linear，用 Sequental(linear, adapter) 替换，并显式命名子模块
            if isinstance(child, nn.Linear):
                input_dim = child.out_features
                seq = nn.Sequential()
                seq.add_module("linear", child)
                seq.add_module("adapter", Adapter(input_dim, adapter_dim, dropout))
                module._modules[name] = seq
        return module

    def load_pretrained_with_patch_adapt(model, ckpt_path, device="cuda"):
        debug_print(f"[INFO] Loading checkpoint {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)

        # === 1️⃣ 计算目标特征尺寸与相对位置长度 ===
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        out = model.image_encoder.patch_embed(dummy)
        target_H, target_W = out.shape[1:3]
        global_target_len = 2 * target_H - 1

        debug_print(f"=== 尺寸适配信息 ===")
        debug_print(f"目标特征图尺寸: ({target_H}, {target_W})")
        debug_print(f"全局相对位置编码长度: {global_target_len}")

        new_state_dict = {}

        for k, v in state_dict.items():
            # ---- patch_embed 卷积核按空间插值 ----
            if "image_encoder.patch_embed.proj.weight" in k:
                old_k, old_c, old_h, old_w = v.shape
                new_k, new_c, new_h, new_w = model.image_encoder.patch_embed.proj.weight.shape
                if (old_h, old_w) != (new_h, new_w):
                    debug_print(f"[ADAPT] {k}: {v.shape} -> {(new_k, new_c, new_h, new_w)}")
                    v = F.interpolate(v, size=(new_h, new_w), mode="bicubic", align_corners=False)
                new_state_dict[k] = v

            # ---- 绝对位置编码 pos_embed 插值到 (target_H, target_W) ----
            elif "image_encoder.pos_embed" in k:
                old_h, old_w = v.shape[1:3]
                if (old_h, old_w) != (target_H, target_W):
                    debug_print(f"[ADAPT] {k}: {v.shape} -> (1,{target_H},{target_W},{v.shape[-1]})")
                    v = v.permute(0, 3, 1, 2)
                    v = F.interpolate(v, size=(target_H, target_W), mode="bicubic", align_corners=False)
                    v = v.permute(0, 2, 3, 1).contiguous()
                new_state_dict[k] = v

            # ---- 跳过相对位置编码（第三步单独处理）----
            elif "rel_pos_h" in k or "rel_pos_w" in k:
                debug_print(f"[SKIP] 跳过 {k}，将在第三步单独处理")
                continue

            else:
                new_state_dict[k] = v

        # === 2️⃣ 加载主要参数（跳过相对位置编码）===
        debug_print("=== 加载主要参数（跳过相对位置编码）===")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        debug_print(f"[INFO] 加载完成: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            debug_print("  missing:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            debug_print("  unexpected:", unexpected[:10], "..." if len(unexpected) > 10 else "")

        # === 3️⃣ 单独处理相对位置编码 ===
        debug_print("=== 单独处理相对位置编码 ===")
        with torch.no_grad():
            for i, blk in enumerate(model.image_encoder.blocks):
                pretrained_h = state_dict.get(f"image_encoder.blocks.{i}.attn.rel_pos_h", None)
                pretrained_w = state_dict.get(f"image_encoder.blocks.{i}.attn.rel_pos_w", None)

                if pretrained_h is None or pretrained_w is None:
                    continue

                # 获取目标 tensor 引用
                current_h = blk.attn.rel_pos_h
                current_w = blk.attn.rel_pos_w

                old_len, dim = pretrained_h.shape
                new_len = current_h.shape[0]  # 模型里目标长度（比如 27 或 63）
                if old_len != new_len:
                    debug_print(f"[ADAPT] Block {i} rel_pos_h: {pretrained_h.shape} -> ({new_len}, {dim})")
                    vv = pretrained_h.transpose(0, 1).unsqueeze(0)
                    vv = F.interpolate(vv, size=new_len, mode='linear', align_corners=False)
                    interpolated_h = vv.squeeze(0).transpose(0, 1).contiguous()
                    blk.attn.rel_pos_h.data.copy_(interpolated_h)
                else:
                    blk.attn.rel_pos_h.data.copy_(pretrained_h)

                old_len, dim = pretrained_w.shape
                new_len = current_w.shape[0]
                if old_len != new_len:
                    debug_print(f"[ADAPT] Block {i} rel_pos_w: {pretrained_w.shape} -> ({new_len}, {dim})")
                    vv = pretrained_w.transpose(0, 1).unsqueeze(0)
                    vv = F.interpolate(vv, size=new_len, mode='linear', align_corners=False)
                    interpolated_w = vv.squeeze(0).transpose(0, 1).contiguous()
                    blk.attn.rel_pos_w.data.copy_(interpolated_w)
                else:
                    blk.attn.rel_pos_w.data.copy_(pretrained_w)

        # === 4️⃣ 验证打印 ===
        debug_print("\n=== 最终验证 ===")
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        features = model.image_encoder.patch_embed(dummy)
        debug_print(f"特征图尺寸: {features.shape}")
        debug_print(f"位置编码尺寸: {model.image_encoder.pos_embed.shape}")

        if len(model.image_encoder.blocks) > 0 and hasattr(model.image_encoder.blocks[0].attn, 'rel_pos_h'):
            rel_pos_h = model.image_encoder.blocks[0].attn.rel_pos_h
            expected_len = 2 * (getattr(model.image_encoder.blocks[0].attn, "window_size", target_H)) - 1
            debug_print(f"Block0 rel_pos_h: {rel_pos_h.shape} (期望长度: {expected_len})")
            debug_print(f"相对位置编码匹配: {rel_pos_h.shape[0] == expected_len}")

        debug_print("[✅] 权重加载与适配完成。模型可安全训练。")
        return model

    # 只打开“选中 block”的 encoder 适配器（Adapter / TransformerAdapter）
    def set_encoder_adapter_trainable(sam, train_layers):
        # 保险：image_encoder 内部全关（可省略，因为上面已经全关了）
        for p in sam.image_encoder.parameters():
            p.requires_grad = False

        # set_encoder_adapter_trainable 内：
        for idx, blk in enumerate(sam.image_encoder.blocks):
            for m in blk.modules():
                if isinstance(m, (Adapter, EncTransformerAdapter)):  # ← 用 EncTransformerAdapter
                    for p in m.parameters():
                        p.requires_grad = (idx in train_layers)

        # blocks 之外统一关也要改同样的 isinstance：
        for name, module in sam.image_encoder.named_modules():
            if "blocks" not in name and isinstance(module, (Adapter, EncTransformerAdapter)):
                for p in module.parameters():
                    p.requires_grad = False

    def build_sam_with_adapter(sam_type="vit_h", sam_ckpt=SAM_CKPT, device=DEVICE,
                               adapter_dim=adapter_dim, adapter_dropout=0.1):
        sam = build_sam_module.sam_model_registry["vit_h"](
            checkpoint=None,
            vit_patch_size=8,
            image_size=256,
            use_adapter=True,
            Drop_path=True,
            adapter_dim=128
        ).to(device)

        sam = load_pretrained_with_patch_adapt(sam, SAM_CKPT, device)

        H_in, W_in = 256, 256  # 或者根据 dataset crop 传入
        dummy = torch.randn(1, 3, H_in, W_in).to(device)
        x = sam.image_encoder.patch_embed(dummy)
        debug_print("patch_embed out:", x.shape, "pos_embed:", sam.image_encoder.pos_embed.shape)

        # patch size
        ps = sam.image_encoder.patch_embed.proj.kernel_size
        debug_print("patch_size:", ps)

        out = sam.image_encoder(torch.randn(1, 3, 256, 256).to(device))
        debug_print("encoder output shape:", out.shape)
        debug_print("pos_embed shape:", sam.image_encoder.pos_embed.shape)

        # 先全关（包含 prompt_encoder / image_encoder / decoders 全部）
        for p in sam.parameters():
            p.requires_grad = False

        # 根据 enc_strategy 选择 enc_train_layers（注意 range 上限要 +1 才包含末端）
        B = 32
        if enc_strategy == "last2":
            enc_train_layers = list(range(B - 2, B))  # [30,31]
        elif enc_strategy == "last4":
            enc_train_layers = list(range(B - 4, B))  # [28,29,30,31]
        elif enc_strategy == "last6":
            enc_train_layers = list(range(B - 6, B))  # [28,29,30,31]
        elif enc_strategy == "sandwich":
            enc_train_layers = [0, 1, 15, 23, B - 2, B - 1]  # [0,1,15,23,30,31]
        elif enc_strategy == "sandwich2":
            enc_train_layers = (0, 5, 11, 17, 22, 23)  # [0,1,15,23,30,31]
        elif enc_strategy == "sandwich32":
            enc_train_layers = list(range(32))
        elif enc_strategy == "sandwich8":
            enc_train_layers = (0, 4, 8, 12, 18, 22, 26, 31)
        elif enc_strategy == "sandwich6":
            enc_train_layers = (0, 3, 7, 20, 27, 31)
        elif enc_strategy == "sandwich6wen":
            enc_train_layers = (0, 4, 8, 12, 16, 23)
        elif enc_strategy in ["auto_top6", "auto_top8"]:
            # 先全开，后面再用梯度重新筛
            enc_train_layers = list(range(B))  # [0..31]
        elif enc_strategy in ["auto_seg6", "auto_seg9"]:
            enc_train_layers = list(range(B))  # 暂时全开，后面再筛
        else:
            raise ValueError(f"unknown enc_strategy: {enc_strategy}")
        set_encoder_adapter_trainable(sam, enc_train_layers)
        # 1) 解冻 patch embedding
        for p in sam.image_encoder.patch_embed.parameters():
            p.requires_grad = True

        # --- 插值位置编码：动态根据 patch_embed 的输出尺寸调整（使用 dataset crop 尺寸）
        # --- 插值位置编码 ---
        with torch.no_grad():
            dummy_size = 256
            dummy = torch.randn(1, 3, dummy_size, dummy_size).to(device)
            x = sam.image_encoder.patch_embed(dummy)  # (1, C, Hf, Wf)
            _, _, Hf, Wf = x.shape
            debug_print("patch_embed out:", x.shape)

            # 原始 pos_embed (1, H0, W0, C)
            x = sam.image_encoder.patch_embed(dummy)  # (1, H, W, C)
            debug_print("patch_embed out:", x.shape)  # (1, 16, 16, 1280)

            # 转为 (B,C,H,W)
            x = x.permute(0, 3, 1, 2).contiguous()
            debug_print("patch_embed permuted:", x.shape)  # (1, 1280, 16, 16)

            _, C, Hf, Wf = x.shape  # ✅ 现在 Hf=16, Wf=16

            # 原始 pos_embed (1, H0, W0, C)
            pos_embed = sam.image_encoder.pos_embed  # (1,64,64,1280)

            # 转为 (B,C,H,W) 插值
            pos_embed_ = pos_embed.permute(0, 3, 1, 2).contiguous()  # (1,1280,64,64)
            pos_embed_resized = F.interpolate(
                pos_embed_,
                size=(Hf, Wf),  # (16,16)
                mode="bicubic",
                align_corners=False
            )  # (1,1280,16,16)

            # 转回 (B,H,W,C) 存储
            pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1).contiguous()  # (1,16,16,1280)
            sam.image_encoder.pos_embed = nn.Parameter(pos_embed_resized.to(device), requires_grad=False)

            debug_print("final pos_embed:", sam.image_encoder.pos_embed.shape)  # (1,16,16,1280)

            # 验证
            x2 = sam.image_encoder.patch_embed(dummy)  # (1,16,16,1280)
            debug_print("x2:", x2.shape, "pos_embed:", sam.image_encoder.pos_embed.shape)

            _ = x2 + sam.image_encoder.pos_embed  # ✅ shape 完全一致
            debug_print("再次 patch_embed out:", x2.shape, "pos_embed:", sam.image_encoder.pos_embed.shape)

        # 4) 构建两个带 Adapter 的 decoder（deepcopy 后默认仍是冻结态）
        sam.mask_decoder_region = add_adapters(copy.deepcopy(sam.mask_decoder),
                                               adapter_dim=adapter_dim,
                                               dropout=adapter_dropout).to(device)
        sam.mask_decoder_boundary = add_adapters(copy.deepcopy(sam.mask_decoder),
                                                 adapter_dim=adapter_dim,
                                                 dropout=adapter_dropout).to(device)
        del sam.mask_decoder  # ← 避免混淆/省显存

        # 5) 只放开 decoder 端 Adapter
        for name, m in sam.named_modules():
            if name.startswith(("mask_decoder_region", "mask_decoder_boundary")):
                if isinstance(m, Adapter):
                    for p in m.parameters():
                        p.requires_grad = True

        total_trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in sam.parameters())
        print(f"构建了带有Adapter的SAM。可训练参数: {total_trainable:,} / {total_all:,}")
        print(
            "Decoder trainable params:",
            sum(p.numel() for p in sam.mask_decoder_region.parameters() if p.requires_grad) +
            sum(p.numel() for p in sam.mask_decoder_boundary.parameters() if p.requires_grad)
        )
        debug_print("image encoder output shape:", sam.image_encoder(torch.randn(1, 3, 256, 256).to(device)).shape)

        # 6) 验证只 adapter 可训练
        trainable_names = [n for n, p in sam.named_parameters() if p.requires_grad]
        allowed_prefix = ["image_encoder.patch_embed", "mask_decoder_region", "mask_decoder_boundary"]

        illegal = []
        for n in trainable_names:
            if ("adapter" not in n) and not any(n.startswith(p) for p in allowed_prefix):
                illegal.append(n)

        assert len(illegal) == 0, f"发现不该解冻的参数: {illegal[:10]}"

        enc_has_adapter = any("image_encoder" in n and "adapter" in n for n, _ in sam.named_parameters())
        dec_has_adapter = any(("mask_decoder_region" in n or "mask_decoder_boundary" in n) and "adapter" in n
                              for n, _ in sam.named_parameters())
        print(f"[VERIFY] encoder_adapter={enc_has_adapter}, decoder_adapter={dec_has_adapter}")
        print(f"[VERIFY] trainable count={sum(p.numel() for p in sam.parameters() if p.requires_grad):,}")
        print("[VERIFY] ==== FINAL trainable parameters ====")
        for n, p in sam.named_parameters():
            if p.requires_grad:
                print(f"{n}: {p.shape}")

        return sam

    def gen_points_and_boxes(prob, device, num_fg=8, num_bg=8):
        # 前景点
        fg = (prob > 0.7).nonzero(as_tuple=False)
        bg = (prob < 0.3).nonzero(as_tuple=False)

        def pick(k, arr):
            if arr.shape[0] == 0: return torch.empty(0, 2, dtype=torch.long, device=device)
            idx = torch.randperm(arr.shape[0], device=device)[:k]
            return arr[idx][:, -2:]

        pts = torch.cat([pick(num_fg, fg), pick(num_bg, bg)], dim=0)  # (N,2)
        labels = torch.cat([
            torch.ones(min(num_fg, fg.shape[0]), dtype=torch.int64, device=device),
            torch.zeros(min(num_bg, bg.shape[0]), dtype=torch.int64, device=device)
        ])

        # 框提示
        mask_bin = (prob[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        ys, xs = np.where(mask_bin > 0)
        if len(xs) > 0 and len(ys) > 0:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            box = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float, device=device)
        else:
            box = None

        return pts, labels, box

    # -----------------------
    # forward function (use prompt encoder + two decoders)
    # -----------------------

    def collate_prompts_padded(pts_list, labels_list, boxes_list, device):
        """
        将每个样本的点(p: [N,2])、label([N]) pad 到 batch 形式，返回 tensors（或 None）。
        pts_tensor: (B, max_N, 2) float, pad value  -1.0
        labels_tensor: (B, max_N) long, pad value -1
        boxes_tensor: (B, max_num_boxes, 4) float  (如果每样本只有 0/1 box，可变成 (B,4))
        """
        B = len(pts_list)
        # 计算最大点数
        max_pts = 0
        for p in pts_list:
            if p is not None:
                max_pts = max(max_pts, int(p.shape[0]))

        if max_pts == 0:
            pts_tensor = None
            labels_tensor = None
        else:
            pts_tensor = torch.full((B, max_pts, 2), -1.0, device=device, dtype=torch.float)
            labels_tensor = torch.full((B, max_pts), -1, device=device, dtype=torch.long)

            for i, (p, l) in enumerate(zip(pts_list, labels_list)):
                if p is None:
                    continue
                # --- 修复维度 ---
                if p.dim() == 3 and p.shape[0] == 1:
                    p = p.squeeze(0)  # (1,N,2) -> (N,2)
                    l = l.squeeze(0)  # (1,N) -> (N,)

                # p: (N,2), l: (N,)
                n = p.shape[0]
                assert pts_tensor[i, :n, :].shape == p.shape, \
                    f"Mismatch: slice={pts_tensor[i, :n, :].shape}, p={p.shape}"

                pts_tensor[i, :n, 0] = p[:, 0].to(device).float()
                pts_tensor[i, :n, 1] = p[:, 1].to(device).float()
                labels_tensor[i, :n] = l.to(device).long()
                debug_print(
                    f"[DEBUG assign] pts_tensor[{i}, :{n}, :].shape={pts_tensor[i, :n, :].shape}, p.shape={p.shape}")
                assert pts_tensor[i, :n, :].shape == p.shape

        # boxes: 如果 boxes_list 中有样本为 None，跳过；支持每样本多个 box（可根据你的数据改）
        if boxes_list is None:
            boxes_tensor = None
        else:
            max_boxes = 0
            for b in boxes_list:
                if b is not None:
                    max_boxes = max(max_boxes, int(b.shape[0]))
            if max_boxes == 0:
                boxes_tensor = None
            else:
                boxes_tensor = torch.zeros((B, max_boxes, 4), device=device, dtype=torch.float)
                for i, b in enumerate(boxes_list):
                    if b is None:
                        raise
                    debug_print(f"[collate] b.shape={b.shape}, 内容={b}")
                    nb = int(b.shape[0])
                    debug_print(f"[collate] nb={nb}, boxes_tensor[{i}, :{nb}, :].shape={boxes_tensor[i, :nb, :].shape}")
                    boxes_tensor[i, :nb, :] = b.to(device).float()

        return pts_tensor, labels_tensor, boxes_tensor

    def forward_two_heads(sam, imgs,
                          pts_r, labels_r, box_r,
                          pts_b, labels_b, box_b,
                          region_prompt, boundary_prompt):
        """
        imgs: (B,3,H,W)
        region_prompt: (B,H,W)  -> 用来提示 region decoder
        boundary_prompt: (B,H,W) -> 用来提示 boundary decoder
        """

        device = next(sam.parameters()).device
        imgs = imgs.to(device)
        # print("[forward_two_heads] region_prompt shapes:", region_prompt.shape)
        # print("[forward_two_heads] boundary_prompt shapes:", boundary_prompt.shape)
        # ========== Encode image ==========
        image_embeddings = sam.image_encoder(imgs)  # (B,C,Hf,Wf)
        # 确保掩膜 prompts 的维度是 (B, 1, H, W) 且是 float 类型 (sam.prompt_encoder 需要)
        # region_prompt 和 boundary_prompt 已经是 long (0/1) 或 bool 类型 [cite: 601]，转为 float 即可
        '''
        if image_embeddings.shape[0] > 1:
            print("Image embedding stats:", image_embeddings[0, 0, 0, 0].item(), image_embeddings[1, 0, 0, 0].item())
        else:
            print("Image embedding stats (only one sample):", image_embeddings[0, 0, 0, 0].item())'''
        #print("image_embeddings mean:", image_embeddings.mean().item())
        #print("region_prompt unique:", torch.unique(region_prompt))
        #print("boundary_prompt unique:", torch.unique(boundary_prompt))

        # =========================
        # ⭐ 统一处理函数（关键）
        # =========================
        def process_prompts(pts, labels, boxes):
            # ---- points ----
            if pts is not None and labels is not None:
                pts = pts.to(device)
                labels = labels.to(device)

                if pts.dim() == 2:  # (N,2) → (1,N,2)
                    pts = pts.unsqueeze(0)

                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)

                points = (pts, labels)
            else:
                points = None

            # ---- boxes ----
            if boxes is not None:
                boxes = boxes.to(device)

                if boxes.dim() == 1:  # (4,) → (1,4)
                    boxes = boxes.unsqueeze(0)

                if boxes.shape[0] != B:
                    boxes = boxes.expand(B, -1)

            return points, boxes

        image_pe = sam.prompt_encoder.get_dense_pe()
        if image_pe.shape[-2:] != image_embeddings.shape[-2:]:
            image_pe = F.interpolate(
                image_pe, size=image_embeddings.shape[-2:], mode="bilinear", align_corners=False
            )

        B = imgs.shape[0]
        pred_r, pred_b = None, None

        # ============================================
        # region head
        # ============================================
        if train_head in ["region", "both"]:
            points_r, boxes_r = process_prompts(pts_r, labels_r, box_r)
            if region_prompt.dim() == 3:
                mask_prompts = region_prompt.unsqueeze(1).float().to(device)
            elif region_prompt.dim() == 4:
                mask_prompts = region_prompt.float().to(device)
            else:
                raise ValueError(f"region_prompt shape {region_prompt.shape} not supported")
            #print("mask_prompts sum:", mask_prompts.sum().item())
            #print("mask_prompts sum:", mask_prompts.sum().item())
            #print("points_r type:", type(points_r))
            '''
            if isinstance(points_r, tuple):
                print("points_r length:", len(points_r))
                for i, item in enumerate(points_r):
                    print(f"  points_r[{i}] type: {type(item)}")
            if points_r is not None:
                pts_tensor, labels_tensor = points_r
                print("points_r[0] sum:", pts_tensor.sum().item())
                print("points_r[0] shape:", pts_tensor.shape)
            '''
            #print("boxes_r sum:", boxes_r.sum().item())
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None,  # points_r, boxes_r
                boxes=None,  # None
                masks=mask_prompts,  # mask_prompts
            )

            low_res_r, _ = sam.mask_decoder_region(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_r = F.interpolate(low_res_r, size=imgs.shape[-2:], mode="bilinear")
            #print(f"pred_r mean: {pred_r.mean().item():.6f}, std: {pred_r.std().item():.6f}")
        # ============================================
        # boundary head
        # ============================================
        if train_head in ["boundary", "both"]:
            points_b, boxes_b = process_prompts(pts_b, labels_b, box_b)
            if boundary_prompt.dim() == 3:
                mask_prompts = boundary_prompt.unsqueeze(1).float().to(device)
            elif boundary_prompt.dim() == 4:
                mask_prompts = boundary_prompt.float().to(device)
            else:
                raise ValueError(f"boundary_prompt shape {boundary_prompt.shape} not supported")
            #print("mask_prompts sum:", mask_prompts.sum().item())
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None,  # points_r, boxes_r
                boxes=None,  # None
                masks=mask_prompts,  # mask_prompts
            )

            low_res_b, _ = sam.mask_decoder_boundary(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            pred_b = F.interpolate(low_res_b, size=imgs.shape[-2:], mode="bilinear")
            #print(f"pred_b mean: {pred_b.mean().item():.6f}, std: {pred_b.std().item():.6f}")

        dummy = torch.zeros((B, 1, imgs.shape[-2], imgs.shape[-1]), device=imgs.device)
        if train_head == "region":
            return pred_r, dummy
        elif train_head == "boundary":
            return dummy, pred_b
        else:
            return pred_r, pred_b

    # -----------------------
    # Linear warmup scheduler (per-iteration): warmup -> constant
    # -----------------------
    def build_warmup_scheduler(optimizer, warmup_steps, base_lr):
        def lr_lambda(step):
            if step + 1 < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def preprocess_prompts(pts_list, labels_list, boxes_list, B, device):
        assert B == len(pts_list) == len(labels_list) == len(boxes_list)
        # ----- points & labels: pad to max_N -----
        max_pts = 0
        for p in pts_list:
            if isinstance(p, torch.Tensor):
                if p.dim() == 3 and p.shape[0] == 1:  # (1,N,2)
                    p = p.squeeze(0)
            if p is not None:
                max_pts = max(max_pts, int(p.shape[-2]))  # (N,2)

        if max_pts == 0:
            pts_tensor = None
            labels_tensor = None
        else:
            pts_tensor = torch.full((B, max_pts, 2), -1.0, device=device, dtype=torch.float)
            labels_tensor = torch.full((B, max_pts), -1, device=device, dtype=torch.long)

            for i, (p, l) in enumerate(zip(pts_list, labels_list)):
                if p is None or l is None:
                    continue
                # squeeze 可能的前导维
                if p.dim() == 3 and p.shape[0] == 1: p = p.squeeze(0)  # (N,2)
                if l.dim() == 2 and l.shape[0] == 1: l = l.squeeze(0)  # (N,)

                n = int(p.shape[0])
                pts_tensor[i, :n, 0] = p[:, 0].to(device).float()
                pts_tensor[i, :n, 1] = p[:, 1].to(device).float()
                labels_tensor[i, :n] = l.to(device).long()

        # ----- boxes: 统一成 (B,4)（单框场景）-----
        boxes_tensor = None
        if boxes_list and any(b is not None for b in boxes_list):
            boxes_tensor = torch.zeros((B, 4), device=device, dtype=torch.float)
            for i, b in enumerate(boxes_list):
                if b is None:
                    continue
                if isinstance(b, torch.Tensor):
                    if b.dim() == 2 and b.shape[0] == 1:  # (1,4)->(4,)
                        b = b.squeeze(0)
                b = b.to(device).float().view(-1)
                if b.numel() >= 4:
                    boxes_tensor[i] = b[:4]

        return pts_tensor, labels_tensor, boxes_tensor

    # -----------------------
    # Training loop (with checkpointing, logs, excel)
    # -----------------------
    def safe_collate(batch):
        # 过滤掉 None 样本
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        return batch

    def train_sam_block_adapter(train_set, val_set, sam_ckpt=SAM_CKPT, epochs=EPOCHS, lr=LR,
                                batch_size=BATCH_SIZE, device=DEVICE, prompter_builder=None,
                                prompter_ckpt=PROMPTER_CKPT):

        # DataLoader: we set batch_size = 1 and accumulate to reach effective batch_size (simpler multi-GPU handling)
        loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=safe_collate)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=safe_collate)

        sam = build_sam_with_adapter(sam_type="vit_h", sam_ckpt=sam_ckpt, device=device)

        # === 自动选 Top-k 梯度敏感层（只在 enc_strategy == "auto_top6" 时启用） ===
        def compute_block_grad_importance(sam, loader, device, max_steps=50):
            """
            返回长度 = num_blocks 的列表，每个元素是该 block 内 Adapter 参数梯度范数的累计值
            """
            sam.train()
            num_blocks = len(sam.image_encoder.blocks)
            scores = [0.0 for _ in range(num_blocks)]

            warmup_dice = DiceLoss()
            warmup_focal = FocalLoss()

            it = iter(loader)
            for step in range(max_steps):
                try:
                    batch = next(it)
                except StopIteration:
                    break

                # === 这部分基本复制你下面训练循环里从 imgs, gt_* 开始到 loss = loss_r + loss_b 结束 ===
                imgs = torch.stack([b[0] for b in batch]).to(device).float()
                gt_region = torch.stack([b[1] for b in batch]).to(device).long()
                gt_boundary = torch.stack([b[2] for b in batch]).to(device).long()

                mask_logits = torch.stack([b[3] for b in batch]).to(device).float()
                region_prompt = torch.stack([b[10] for b in batch]).to(device).float()  # (B,1,H,W) or (B,H,W)
                boundary_prompt = torch.stack([b[11] for b in batch]).to(device).float()

                pts_list_r, labels_list_r, boxes_list_r = [], [], []
                pts_list_b, labels_list_b, boxes_list_b = [], [], []
                for b in batch:
                    p_r = b[4];
                    l_r = b[5];
                    bx_r = b[6]
                    p_b = b[7];
                    l_b = b[8];
                    bx_b = b[9]
                    if isinstance(p_r, torch.Tensor) and p_r.dim() == 3 and p_r.shape[0] == 1:
                        p_r = p_r.squeeze(0)
                    if isinstance(l_r, torch.Tensor) and l_r.dim() == 2 and l_r.shape[0] == 1:
                        l_r = l_r.squeeze(0)
                    if isinstance(bx_r, torch.Tensor) and bx_r.dim() == 2 and bx_r.shape[0] == 1:
                        bx_r = bx_r.squeeze(0)
                    if isinstance(bx_r, torch.Tensor) and bx_r.dim() == 1:
                        bx_r = bx_r.unsqueeze(0)
                    pts_list_r.append(p_r.to(device))
                    labels_list_r.append(l_r.to(device))
                    boxes_list_r.append(bx_r.to(device))
                    if isinstance(p_b, torch.Tensor) and p_b.dim() == 3 and p_b.shape[0] == 1:
                        p_b = p_b.squeeze(0)
                    if isinstance(l_b, torch.Tensor) and l_b.dim() == 2 and l_b.shape[0] == 1:
                        l_b = l_b.squeeze(0)
                    if isinstance(bx_b, torch.Tensor) and bx_b.dim() == 2 and bx_b.shape[0] == 1:
                        bx_b = bx_b.squeeze(0)
                    if isinstance(bx_b, torch.Tensor) and bx_b.dim() == 1:
                        bx_b = bx_b.unsqueeze(0)
                    pts_list_b.append(p_b.to(device))
                    labels_list_b.append(l_b.to(device))
                    boxes_list_b.append(bx_b.to(device))

                    # 将列表转换为批处理 tensor
                    pts_r_batch, labels_r_batch, boxes_r_batch = preprocess_prompts(
                        pts_list_r, labels_list_r, boxes_list_r, B=imgs.shape[0], device=device
                    )
                    pts_b_batch, labels_b_batch, boxes_b_batch = preprocess_prompts(
                        pts_list_b, labels_list_b, boxes_list_b, B=imgs.shape[0], device=device
                    )

                    pred_r, pred_b = forward_two_heads(
                        sam, imgs,
                        pts_r_batch, labels_r_batch, boxes_r_batch,
                        pts_b_batch, labels_b_batch, boxes_b_batch,
                        region_prompt, boundary_prompt
                    )

                pred_r_sig = torch.sigmoid(pred_r)
                pred_b_sig = torch.sigmoid(pred_b)

                gt_region_resized = F.interpolate(
                    gt_region.unsqueeze(1).float(),
                    size=pred_r_sig.shape[-2:],
                    mode="nearest"
                ).squeeze(1).long()

                gt_boundary_resized = F.interpolate(
                    gt_boundary.unsqueeze(1).float(),
                    size=pred_b_sig.shape[-2:],
                    mode="nearest"
                ).squeeze(1).long()

                dice_r_loss = warmup_dice(pred_r_sig, gt_region_resized, reduction='mean')
                focal_r_loss = warmup_focal(pred_r, gt_region_resized, reduction='mean')
                dice_b_loss = warmup_dice(pred_b_sig, gt_boundary_resized, reduction='mean')
                focal_b_loss = warmup_focal(pred_b, gt_boundary_resized, reduction='mean')

                if a == 0:
                    loss_r = focal_r_loss
                    loss_b = focal_b_loss
                elif a == "mean":
                    dice_r_per_sample = warmup_dice(pred_r_sig, gt_region_resized, reduction='none')
                    focal_r_per_sample = warmup_focal(pred_r, gt_region_resized, reduction='none')
                    dice_b_per_sample = warmup_dice(pred_b_sig, gt_boundary_resized, reduction='none')
                    focal_b_per_sample = warmup_focal(pred_b, gt_boundary_resized, reduction='none')

                    norm_r_dice = dice_r_per_sample / (dice_r_per_sample.mean().detach() + 1e-8)
                    norm_r_focal = focal_r_per_sample / (focal_r_per_sample.mean().detach() + 1e-8)
                    loss_r = (norm_r_dice + norm_r_focal).mean()

                    norm_b_dice = dice_b_per_sample / (dice_b_per_sample.mean().detach() + 1e-8)
                    norm_b_focal = focal_b_per_sample / (focal_b_per_sample.mean().detach() + 1e-8)
                    loss_b = (norm_b_dice + norm_b_focal).mean()
                elif a == "sum":
                    dice_r_per_sample = warmup_dice(pred_r_sig, gt_region_resized, reduction='none')
                    focal_r_per_sample = warmup_focal(pred_r, gt_region_resized, reduction='none')
                    dice_b_per_sample = warmup_dice(pred_b_sig, gt_boundary_resized, reduction='none')
                    focal_b_per_sample = warmup_focal(pred_b, gt_boundary_resized, reduction='none')

                    scale_r = dice_r_per_sample.detach() + focal_r_per_sample.detach() + 1e-8
                    scale_b = dice_b_per_sample.detach() + focal_b_per_sample.detach() + 1e-8

                    norm_r_dice = dice_r_per_sample / scale_r
                    norm_r_focal = focal_r_per_sample / scale_r
                    loss_r = (norm_r_dice + norm_r_focal).mean()

                    norm_b_dice = dice_b_per_sample / scale_b
                    norm_b_focal = focal_b_per_sample / scale_b
                    loss_b = (norm_b_dice + norm_b_focal).mean()
                elif a == "动态权重":
                    scale_r = dice_r_loss.detach() + focal_r_loss.detach() + 1e-8
                    weight_dice_r = focal_r_loss.detach() / scale_r
                    weight_focal_r = dice_r_loss.detach() / scale_r
                    loss_r = weight_dice_r * dice_r_loss + weight_focal_r * focal_r_loss

                    scale_b = dice_b_loss.detach() + focal_b_loss.detach() + 1e-8
                    weight_dice_b = focal_b_loss.detach() / scale_b
                    weight_focal_b = dice_b_loss.detach() / scale_b
                    loss_b = weight_dice_b * dice_b_loss + weight_focal_b * focal_b_loss
                elif a == 0.4:
                    lambda_focal = 5.0
                    loss_r = a * dice_r_loss + (1 - a) * lambda_focal * focal_r_loss
                    loss_b = a * dice_b_loss + (1 - a) * lambda_focal * focal_b_loss

                if train_head == "region":
                    loss = loss_r
                elif train_head == "boundary":
                    loss = loss_b
                elif train_head == "both":
                    loss = loss_r + loss_b

                sam.zero_grad(set_to_none=True)
                loss.backward()

                # === 累积每个 block 内 Adapter 的梯度范数 ===
                for idx, blk in enumerate(sam.image_encoder.blocks):
                    total = 0.0
                    for m in blk.modules():
                        if isinstance(m, (Adapter, EncTransformerAdapter)):
                            for p in m.parameters():
                                if p.grad is not None:
                                    g = p.grad.detach()
                                    total += g.norm(2).item() ** 2
                    scores[idx] += total ** 0.5

            sam.zero_grad(set_to_none=True)
            return scores

        def segmented_topk(block_scores, segments):
            """
            block_scores: list[float], len=B
            segments: list of tuples (start, end, k)  # end 为开区间
                      e.g. [(0, 11, 2), (11, 22, 2), (22, 32, 2)]
            return: sorted list of selected layer indices
            """
            scores = np.array(block_scores)
            picked = []
            for (s, e, k) in segments:
                part = scores[s:e]
                if len(part) == 0:
                    continue
                # 从大到小取 top-k（在该段内）
                local_idx = np.argsort(part)[::-1][:k]
                picked.extend((local_idx + s).tolist())
            # 去重 + 排序
            return sorted(list(dict.fromkeys(picked)))

        k = 8
        # 如果是自动策略，用梯度敏感性选 Top-6 层，再重新设置可训练层
        if enc_strategy == f"auto_top{k}":
            print(f"[AUTO] enc_strategy = auto_top{k}，开始梯度敏感性 warmup ...")
            block_scores = compute_block_grad_importance(sam, loader, device, max_steps=100)

            import numpy as _np
            indices = _np.argsort(_np.array(block_scores))[::-1]  # 从大到小
            topk_layers = sorted(indices[:k].tolist())
            print(f"[AUTO] 选出的 Top-{k} encoder blocks:", topk_layers)
            # 打印32层的排序
            rank = np.argsort(np.array(block_scores))[::-1]  # 从大到小
            print("\n[AUTO] Encoder 32层敏感度排序（从高到低）:")
            for i, layer_id in enumerate(rank):
                print(f"Rank {i + 1:02d}: Block {layer_id}  Score={block_scores[layer_id]:.6f}")

            print("[AUTO] 每层累计梯度范数：", block_scores)
            # ===== 保存结果到Excel =====
            # 将结果保存到Excel表
            # 创建DataFrame
            df = pd.DataFrame({
                'Rank': list(range(1, 33)),  # 排名1-32
                'Layer_ID': rank.tolist(),  # 层ID（按敏感度排序）
                'Layer_Score': [block_scores[layer_id] for layer_id in rank],  # 对应分数
                'Original_Layer_Index': list(range(32)),  # 原始层索引 0-31
                'Original_Score': block_scores  # 原始分数（按层索引顺序）
            })
            # 确保OUT_DIR存在
            if not os.path.exists(OUT_DIR):
                os.makedirs(OUT_DIR)
            # 保存为Excel文件
            excel_path = os.path.join(OUT_DIR, 'layer_sensitivity_analysis.xlsx')
            df.to_excel(excel_path, index=False)
            print(f"\n[AUTO] 结果已保存到: {excel_path}")
        # 如果是自动策略，用梯度敏感性选 Top-6 层，再重新设置可训练层
        if enc_strategy == "auto_seg6" or enc_strategy == "auto_seg9":
            print("[AUTO] enc_strategy = auto_seg6，开始梯度敏感性 warmup ...")
            block_scores = compute_block_grad_importance(sam, loader, device, max_steps=100)

            # ===== 分段 Top-6（浅/中/深 各 2 层） =====
            B = len(block_scores)  # 32
            segments = [
                (0, 11, 2),  # 浅层 0-10 选 2
                (11, 22, 2),  # 中层 11-21 选 2
                (22, 32, 2)  # 深层 22-31 选 2
            ]
            topk_layers = segmented_topk(block_scores, segments)

            print("[AUTO] 每层累计梯度范数：", block_scores)
            print(f"[AUTO] 分段 Top-6 encoder blocks:", topk_layers)

            # 用公共函数重新设置 encoder 哪些层的 Adapter 可训练
            set_encoder_adapter_trainable(sam, topk_layers)
            # 打印32层的排序
            rank = np.argsort(np.array(block_scores))[::-1]  # 从大到小
            print("\n[AUTO] Encoder 32层敏感度排序（从高到低）:")
            for i, layer_id in enumerate(rank):
                print(f"Rank {i + 1:02d}: Block {layer_id}  Score={block_scores[layer_id]:.6f}")

            print("[AUTO] 每层累计梯度范数：", block_scores)
            # ===== 保存结果到Excel =====
            # 将结果保存到Excel表
            # 创建DataFrame
            df = pd.DataFrame({
                'Rank': list(range(1, 33)),  # 排名1-32
                'Layer_ID': rank.tolist(),  # 层ID（按敏感度排序）
                'Layer_Score': [block_scores[layer_id] for layer_id in rank],  # 对应分数
                'Original_Layer_Index': list(range(32)),  # 原始层索引 0-31
                'Original_Score': block_scores  # 原始分数（按层索引顺序）
            })
            # 确保OUT_DIR存在
            if not os.path.exists(OUT_DIR):
                os.makedirs(OUT_DIR)
            # 保存为Excel文件
            excel_path = os.path.join(OUT_DIR, 'layer_sensitivity_analysis.xlsx')
            df.to_excel(excel_path, index=False)
            print(f"\n[AUTO] 结果已保存到: {excel_path}")

        PRINT_EVERY = 50  # 每隔多少 step 打印一次

        def print1(flag, step=None, *args, every=None, **kwargs):
            if not flag:
                return print(*args, **kwargs)
            if step is None:
                return print(*args, **kwargs)
            n = PRINT_EVERY if every is None else int(every)
            if n <= 0 or (step % n == 0):
                return print(*args, **kwargs)
            return None

        # ✅ 不要重复设置 requires_grad
        # 直接使用 build_sam_with_adapter 内返回的可训练参数
        trainable_params = [p for p in sam.parameters() if p.requires_grad]
        print1(False,
               f"[FIX] 使用 build_sam_with_adapter 内部设定的 adapter 参数，共 {sum(p.numel() for p in trainable_params):,} 个")
        for idx, blk in enumerate(sam.image_encoder.blocks):
            for name, module in blk.named_modules():
                if "adapter" in name:
                    print(f"Block {idx} adapter found: {name}")
        print(len(sam.image_encoder.blocks))


        # 1) 解冻 patch embedding
        # patch 永远训
        for p in sam.image_encoder.patch_embed.parameters():
            p.requires_grad = True
        # 2) 确保decoder中非adapter部分冻结，adapter部分可训练
        for n, p in sam.named_parameters():
            # decoder adapter可训练
            if ("mask_decoder_region" in n or "mask_decoder_boundary" in n) and "adapter" in n:
                p.requires_grad = True
            # decoder非adapter部分冻结
            elif ("mask_decoder_region" in n or "mask_decoder_boundary" in n) and "adapter" not in n:
                p.requires_grad = False
        # 根据 train_head 冻结对应的整个解码器（包括其 Adapter）
        if train_head == "region":
            print("🔒 冻结 Boundary Decoder")
            for p in sam.mask_decoder_boundary.parameters():
                p.requires_grad = False
        elif train_head == "boundary":
            print("🔒 冻结 Region Decoder")
            for p in sam.mask_decoder_region.parameters():
                p.requires_grad = False

        total_trainable_after = sum(p.numel() for p in sam.parameters() if p.requires_grad)
        print(f"冻结后实际可训练参数: {total_trainable_after:,}")
        # 3) 收集 encoder adapter 参数
        enc_params = []
        patch_params = [p for p in sam.image_encoder.patch_embed.parameters() if p.requires_grad]
        enc_params = [
            p for n, p in sam.named_parameters()
            if p.requires_grad and "image_encoder" in n and "adapter" in n
        ]
        # ========== (B) Decoder解冻 =============
        # ✅ 只收集decoder中的adapter参数
        dec_params = [
            p for n, p in sam.named_parameters()
            if p.requires_grad and ("mask_decoder_region" in n or "mask_decoder_boundary" in n) and "adapter" in n
        ]
        print(f"[LR-GROUP] encoder adapters: {len(enc_params)} params")
        print(f"[LR-GROUP] decoder adapters: {len(dec_params)}")

        '''optimizer = torch.optim.AdamW([
            {"params": enc_params, "lr": 8e-5, "weight_decay": 1e-2},  #5e-5 5e-3
            {"params": dec_params, "lr": 2e-4, "weight_decay": 1e-2},  #1e-4 5e-3
        ])'''
        optimizer = torch.optim.AdamW([
            {"params": patch_params, "lr": 1e-5, "weight_decay": 0.0},  # patch embedding 微调最稳
            {"params": enc_params, "lr": 8e-5, "weight_decay": 1e-2},  # encoder adapters
            {"params": dec_params, "lr": 1e-4, "weight_decay": 1e-2},  # decoder adapters
        ])

        print(f"[PARAMS] patch: {len(patch_params)}")
        print(f"[PARAMS] enc_adapter: {len(enc_params)}")
        print(f"[PARAMS] dec_adapter: {len(dec_params)}")

        for n, p in sam.named_parameters():
            if 'adapter' in n:
                print1(False, f"[VERIFY] {n} | requires_grad={p.requires_grad} | id={id(p)}")
                print1(False, f"{n}: requires_grad={p.requires_grad}, mean={p.data.mean().item():.6f}")
        for i, g in enumerate(optimizer.param_groups[0]['params']):
            print1(False, f"[OPT_PARAM] idx={i}, id={id(g)}")

        print1(False, ">>> 检查 sam 是否被重新封装:")
        print1(False, "type(sam):", type(sam))
        if hasattr(sam, 'module'):
            print1(False, "sam.module 存在，实际训练的模型是 DataParallel 封装的")

        # === AMP: mixed precision setup (new API, PyTorch ≥2.0) ===
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        from torch.cuda.amp import GradScaler
        from torch import amp
        scaler = GradScaler()

        # === end AMP ===

        scheduler = build_warmup_scheduler(optimizer, warmup_steps=WARMUP_STEPS, base_lr=lr)

        # === 构建 prompter ===
        assert prompter_builder is not None, "需要提供 prompter_builder"
        prompter = prompter_builder().to(device)
        ckpt = torch.load(prompter_ckpt, map_location=device)
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        # ===== 修正 key 前缀 =====
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("net."):
                new_k = "model." + k[4:]  # net.xxx -> model.xxx
            else:
                new_k = k
            new_state_dict[new_k] = v
        missing, unexpected = prompter.load_state_dict(new_state_dict)
        print("missing:", missing)
        print("unexpected:", unexpected)
        prompter.eval()

        dice = DiceLoss();
        focal = FocalLoss()

        # checkpoint support
        early_stopping = EarlyStopping(patience=patience, path=best_path)
        start_epoch = 0
        global_step = 0
        logs = []  # 验证集日志
        train_logs = []  # 训练集日志
        # 初始化阈值变量为默认值
        best_thr_region = 0.5
        best_thr_boundary = 0.5
        best_region_metrics = {"IoU": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}
        best_boundary_metrics = {"IoU": 0.0, "F1": 0.0, "Precision": 0.0, "Recall": 0.0}
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device)
            sam.load_state_dict(ck["model"], strict=False)
            optimizer.load_state_dict(ck["optimizer"])
            try:
                scheduler.load_state_dict(ck["scheduler"])
            except Exception:
                pass
            if "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
            else:
                print("Warning: no scaler state found; starting AMP scaler from default.")

            start_epoch = ck.get("epoch", 0)
            global_step = ck.get("global_step", 0)

            # [新增/修改] 2. 尝试从断点中加载最佳阈值
            if "best_thr_region" in ck:
                best_thr_region = ck["best_thr_region"]
            if "best_thr_boundary" in ck:
                best_thr_boundary = ck["best_thr_boundary"]

                # [新增] 尝试加载上次保存的最佳指标
            if "best_region_metrics" in ck:
                best_region_metrics = ck["best_region_metrics"]
            if "best_boundary_metrics" in ck:
                best_boundary_metrics = ck["best_boundary_metrics"]

            # [新增/修改] 3. 双重保险：如果断点里没有（旧代码生成的），尝试从 best_model 里读
            if ("best_thr_region" not in ck) and os.path.exists(best_path):
                try:
                    ck_best = torch.load(best_path, map_location=device)
                    if "best_thr_region" in ck_best:
                        best_thr_region = ck_best["best_thr_region"]
                    if "best_thr_boundary" in ck_best:
                        best_thr_boundary = ck_best["best_thr_boundary"]
                    print(f" ℹ️ 从最佳模型补录阈值: Region={best_thr_region}, Boundary={best_thr_boundary}")
                except Exception as e:
                    print(f" ⚠️ 无法从最佳模型加载阈值: {e}")
            print(f"恢复训练: 从 epoch {start_epoch} , global_step {global_step}")
            if "early_stopping" in ck:
                es_state = ck["early_stopping"]
                early_stopping.best_score = es_state.get("best_score", -float("inf"))
                early_stopping.counter = es_state.get("counter", 0)
                early_stopping.early_stop = es_state.get("early_stop", False)

        sam.train()
        total_start = time.time()

        for ep in range(start_epoch, epochs):
            pbar = tqdm(loader, desc=f"训练 Epoch {ep + 1}/{epochs}", ncols=200)
            acc_step = 0
            accum_loss = 0.0
            window_loss_sum = 0.0
            window_count = 0
            steps_done = 0
            train_tp_r = train_fp_r = train_fn_r = 0
            train_tp_b = train_fp_b = train_fn_b = 0
            train_loss_sum = 0

            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(pbar):
                imgs = torch.stack([b[0] for b in batch]).to(device).float()
                gt_region = torch.stack([b[1] for b in batch]).to(device).long()
                gt_boundary = torch.stack([b[2] for b in batch]).to(device).long()
                region_prompt = torch.stack([b[10] for b in batch]).to(device).long()
                boundary_prompt = torch.stack([b[11] for b in batch]).to(device).long()

                mask_logits = torch.stack([b[3] for b in batch]).to(device).float()
                # ---- robust prompt preprocessing for training ----
                pts_list_r = []
                labels_list_r = []
                boxes_list_r = []
                pts_list_b = []
                labels_list_b = []
                boxes_list_b = []

                for bidx, b in enumerate(batch):
                    p_r = b[4]
                    l_r = b[5]
                    bx_r = b[6]
                    p_b = b[7]
                    l_b = b[8]
                    bx_b = b[9]
                    # 如果形状是 (1, N, 2) -> squeeze -> (N,2)
                    if isinstance(p_r, torch.Tensor) and p_r.dim() == 3 and p_r.shape[0] == 1:
                        p_r = p.squeeze(0)
                    # 如果形状是 (N,2) 好的
                    if isinstance(l_r, torch.Tensor) and l_r.dim() == 2 and l_r.shape[0] == 1:
                        l_r = l_r.squeeze(0)
                    # boxes: 有时是 (1,4) 或 (4,)
                    if isinstance(bx_r, torch.Tensor) and bx_r.dim() == 2 and bx_r.shape[0] == 1:
                        bx_r = bx_r.squeeze(0)
                    # 如果是单个 box (4,) -> unsqueeze -> (1,4)
                    if isinstance(bx_r, torch.Tensor) and bx_r.dim() == 1:
                        bx_r = bx_r.unsqueeze(0)
                    pts_list_r.append(p_r.to(device))
                    labels_list_r.append(l_r.to(device))
                    boxes_list_r.append(bx_r.to(device))
                    # 如果形状是 (1, N, 2) -> squeeze -> (N,2)
                    if isinstance(p_b, torch.Tensor) and p_b.dim() == 3 and p_b.shape[0] == 1:
                        p_b = p.squeeze(0)
                    # 如果形状是 (N,2) 好的
                    if isinstance(l_b, torch.Tensor) and l_b.dim() == 2 and l_b.shape[0] == 1:
                        l_b = l_b.squeeze(0)
                    # boxes: 有时是 (1,4) 或 (4,)
                    if isinstance(bx_b, torch.Tensor) and bx_b.dim() == 2 and bx_b.shape[0] == 1:
                        bx_b = bx_b.squeeze(0)
                    # 如果是单个 box (4,) -> unsqueeze -> (1,4)
                    if isinstance(bx_b, torch.Tensor) and bx_b.dim() == 1:
                        bx_b = bx_b.unsqueeze(0)
                    pts_list_b.append(p_b.to(device))
                    labels_list_b.append(l_b.to(device))
                    boxes_list_b.append(bx_b.to(device))
                # 可选 debug（把下面两行打开以便查看每个 batch 的 shapes）
                debug_print("[TRAIN] pts_list_r shapes:", [getattr(x, 'shape', None) for x in pts_list_r])
                debug_print("[TRAIN] boxes_list_r shapes:", [getattr(x, 'shape', None) for x in boxes_list_r])
                debug_print("[TRAIN] pts_list_b shapes:", [getattr(x, 'shape', None) for x in pts_list_b])
                debug_print("[TRAIN] boxes_list_b shapes:", [getattr(x, 'shape', None) for x in boxes_list_b])
                # === AMP forward + backward ===
                try:
                    with amp.autocast('cuda'):
                        # 将列表转换为批处理 tensor
                        pts_r_batch, labels_r_batch, boxes_r_batch = preprocess_prompts(
                            pts_list_r, labels_list_r, boxes_list_r, B=imgs.shape[0], device=device
                        )
                        pts_b_batch, labels_b_batch, boxes_b_batch = preprocess_prompts(
                            pts_list_b, labels_list_b, boxes_list_b, B=imgs.shape[0], device=device
                        )

                        pred_r, pred_b = forward_two_heads(
                            sam, imgs,
                            pts_r_batch, labels_r_batch, boxes_r_batch,
                            pts_b_batch, labels_b_batch, boxes_b_batch,
                            region_prompt, boundary_prompt
                        )
                        # logits (B,1,h,w) — sigmoids 可以放在 autocast 内或外都行
                        pred_r_sig = torch.sigmoid(pred_r)
                        pred_b_sig = torch.sigmoid(pred_b)

                        # 上采样 ground truth 到预测分辨率
                        gt_region_resized = F.interpolate(
                            gt_region.unsqueeze(1).float(),
                            size=pred_r_sig.shape[-2:],
                            mode="nearest"
                        ).squeeze(1).long()

                        gt_boundary_resized = F.interpolate(
                            gt_boundary.unsqueeze(1).float(),
                            size=pred_b_sig.shape[-2:],
                            mode="nearest"
                        ).squeeze(1).long()

                        # 计算损失（尽量把数学运算放在 autocast 内以节省显存）
                        dice_r_loss = dice(pred_r_sig, gt_region_resized, reduction='mean')
                        focal_r_loss = focal(pred_r, gt_region_resized, reduction='mean')

                        dice_b_loss = dice(pred_b_sig, gt_boundary_resized, reduction='mean')
                        focal_b_loss = focal(pred_b, gt_boundary_resized, reduction='mean')
                        dice_r_per_sample = dice(pred_r_sig, gt_region_resized, reduction='none')  # [B]
                        focal_r_per_sample = focal(pred_r, gt_region_resized, reduction='none')  # [B]

                        dice_b_per_sample = dice(pred_b_sig, gt_boundary_resized, reduction='none')  # [B]
                        focal_b_per_sample = focal(pred_b, gt_boundary_resized, reduction='none')  # [B]
                        if a == 0:
                            # 只用 FocalLoss
                            loss_r = focal_r_loss
                            loss_b = focal_b_loss
                        elif a == "mean":
                            # R 通道
                            norm_r_dice = dice_r_per_sample / (dice_r_per_sample.mean().detach() + 1e-8)
                            norm_r_focal = focal_r_per_sample / (focal_r_per_sample.mean().detach() + 1e-8)
                            loss_r = (norm_r_dice + norm_r_focal).mean()

                            # B 通道
                            norm_b_dice = dice_b_per_sample / (dice_b_per_sample.mean().detach() + 1e-8)
                            norm_b_focal = focal_b_per_sample / (focal_b_per_sample.mean().detach() + 1e-8)
                            loss_b = (norm_b_dice + norm_b_focal).mean()
                        elif a == "sum":
                            # 计算每个样本的scale
                            scale_r_per_sample = dice_r_per_sample.detach() + focal_r_per_sample.detach() + 1e-8

                            norm_r_dice = dice_r_per_sample / scale_r_per_sample
                            norm_r_focal = focal_r_per_sample / scale_r_per_sample

                            loss_r = (norm_r_dice + norm_r_focal).mean()

                            # Boundary 同理
                            scale_b_per_sample = dice_b_per_sample.detach() + focal_b_per_sample.detach() + 1e-8

                            norm_b_dice = dice_b_per_sample / scale_b_per_sample
                            norm_b_focal = focal_b_per_sample / scale_b_per_sample

                            loss_b = (norm_b_dice + norm_b_focal).mean()
                        elif a == "动态权重":
                            scale_r = dice_r_loss.detach() + focal_r_loss.detach() + 1e-8
                            weight_dice_r = focal_r_loss.detach() / scale_r
                            weight_focal_r = dice_r_loss.detach() / scale_r
                            loss_r = weight_dice_r * dice_r_loss + weight_focal_r * focal_r_loss

                            scale_b = dice_b_loss.detach() + focal_b_loss.detach() + 1e-8
                            weight_dice_b = focal_b_loss.detach() / scale_b
                            weight_focal_b = dice_b_loss.detach() / scale_b
                            loss_b = weight_dice_b * dice_b_loss + weight_focal_b * focal_b_loss
                        elif a == 0.4:
                            lambda_focal = 5.0
                            loss_r = a * dice_r_loss + (1 - a) * lambda_focal * focal_r_loss
                            loss_b = a * dice_b_loss + (1 - a) * lambda_focal * focal_b_loss

                        if train_head == "region":
                            loss = loss_r
                        elif train_head == "boundary":
                            loss = loss_b
                        elif train_head == "both":
                            loss = loss_r + loss_b

                        train_loss_sum += loss.item()
                        # ===== region metrics =====
                        pred_r = (pred_r_sig > 0.5).float()

                        tp = ((pred_r == 1) & (gt_region_resized == 1)).sum().item()
                        fp = ((pred_r == 1) & (gt_region_resized == 0)).sum().item()
                        fn = ((pred_r == 0) & (gt_region_resized == 1)).sum().item()

                        train_tp_r += tp
                        train_fp_r += fp
                        train_fn_r += fn

                        # ===== boundary metrics =====
                        pred_b = (pred_b_sig > 0.5).float()

                        tp = ((pred_b == 1) & (gt_boundary_resized == 1)).sum().item()
                        fp = ((pred_b == 1) & (gt_boundary_resized == 0)).sum().item()
                        fn = ((pred_b == 0) & (gt_boundary_resized == 1)).sum().item()

                        train_tp_b += tp
                        train_fp_b += fp
                        train_fn_b += fn
                        # 监控（更安全地 detach 到 cpu）
                        if step % 50 == 0:
                            pr_mean = float(pred_r_sig.detach().cpu().mean().item())
                            pb_mean = float(pred_b_sig.detach().cpu().mean().item())
                            print1(True, step,
                                   f"[Monitor] Step {step} | pred_r_sig.mean={pr_mean:.4f}, pred_b_sig.mean={pb_mean:.4f} | loss_r={loss_r.mean().item():.4f}, loss_b={loss_b.mean().item():.4f}")

                except Exception as e:
                    debug1_print(f"[ERROR train] forward_two_heads failed on batch idx with error: {e}")
                    debug1_print("  imgs.shape:", getattr(imgs, "shape", None))
                    debug1_print("  pts_list_r shapes:", [getattr(x, "shape", None) for x in pts_list_r])
                    debug1_print("  labels_list_r shapes:", [getattr(x, "shape", None) for x in labels_list_r])
                    debug1_print("  boxes_list_r shapes:", [getattr(x, "shape", None) for x in boxes_list_r])
                    debug1_print("  pts_list_b shapes:", [getattr(x, "shape", None) for x in pts_list_b])
                    debug1_print("  labels_list_b shapes:", [getattr(x, "shape", None) for x in labels_list_b])
                    debug1_print("  boxes_list_b shapes:", [getattr(x, "shape", None) for x in boxes_list_b])
                    raise

                # ---- 在 loss 计算之后，backward 之前：打印 lr/list of adapter params ----
                # print optimizer & adapter info
                print1(True, step, "[DBG] optimizer param_groups lr:", [pg['lr'] for pg in optimizer.param_groups])
                adapter_params = [(n, p) for n, p in sam.named_parameters() if 'adapter' in n]
                print1(True, step, "[DBG] adapter param count:", len(adapter_params))
                for n, p in adapter_params[:3]:  # 打前三个查看
                    print1(True, step,
                           f"[DBG] adapter name={n}, requires_grad={p.requires_grad}, mean={p.data.mean().item():.8e}")

                # ---- 检查 loss 是否参与计算图（有无 grad_fn） ----
                print1(True, step, "[DBG] loss grad_fn:", getattr(loss, "grad_fn", None))

                # —— 在创建 optimizer 之后（确保 adapter 的 requires_grad 已设为 True） ——

                # 运行 1 个 batch 后在 loss.backward() 前后检查
                print1(True, step, "[DBG_TEST] optimizer lr:", [pg['lr'] for pg in optimizer.param_groups])

                adapter_params = [(n, p) for n, p in sam.named_parameters() if 'adapter' in n and p.requires_grad]
                print1(True, step, "[DBG_TEST] adapter param count:", len(adapter_params))
                for n, p in adapter_params[:3]:
                    print1(True, step,
                           f"[DBG_TEST] before: {n}, requires_grad={p.requires_grad}, mean={p.data.mean().item():.8e}")

                print1(True, step, "[DBG_TEST] loss grad_fn:", getattr(loss, "grad_fn", None))

                # backward
                scaler.scale(loss / accum_steps).backward()

                # 检查 grads
                none_count = sum(1 for _, p in adapter_params if p.grad is None)
                zero_count = sum(1 for _, p in adapter_params if (p.grad is not None and p.grad.norm().item() == 0.0))
                print1(True, step, f"[DBG_TEST] adapter grads: None={none_count}, zero_norm={zero_count}")
                for n, p in adapter_params[:3]:
                    print1(True, step,
                           f"[DBG_TEST] grad for {n}: is None? {p.grad is None}, norm={(0.0 if p.grad is None else p.grad.norm().item()):.6e}")

                # record one param, step, check change
                will_step = ((step + 1) % accum_steps == 0)
                do_debug_record = (step % PRINT_EVERY == 0) and len(adapter_params) > 0
                do_debug = will_step and do_debug_record
                if len(adapter_params) == 0:
                    print1(True, step, "[WARN] adapter_params empty -> skip debug checks for this batch")

                if do_debug_record:
                    before = adapter_params[0][1].data.clone()

                if (step + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        # 只有在优化器真的更新过之后才允许 scheduler.step()
                        scheduler.step()
                    first_opt_step_done = True
                    optimizer.zero_grad(set_to_none=True)
                    print1(True, step, "[LR now]:", scheduler.get_last_lr())
                    if global_step < 10:
                        print1(True, step,
                               "[LR DEBUG]",
                               [g["lr"] for g in optimizer.param_groups]
                               )

                if do_debug:
                    after = adapter_params[0][1].data
                    print1(True, step,
                           f"[DBG_TEST] adapter param mean before={before.mean().item():.8e}, after={after.mean().item():.8e}, diff={(after - before).abs().mean().item():.8e}")

                # 记录/打印
                accum_loss += loss.item()  # 注意 loss 在 autocast 内是 Scale 后的 tensor? .item() 自动反标量
                acc_step += 1
                global_step += 1
                dice_r_train = dice_r_loss.item()
                focal_r_train = focal_r_loss.item()
                dice_b_train = dice_b_loss.item()
                focal_b_train = focal_b_loss.item()
                loss_r_val = loss_r.item()
                loss_b_val = loss_b.item()

                # 🔑 先定义默认 IoU
                batch_iou = 0.0

                # 计算当前 batch IoU
                with torch.no_grad():
                    preds = (pred_r_sig > 0.5).long().cpu().numpy()
                    gts = gt_region_resized.cpu().numpy()

                    if preds.size > 0:  # 防止空
                        batch_ious = []
                        for i in range(preds.shape[0]):
                            m = compute_pixel_metrics(gts[i], preds[i, 0])
                            batch_ious.append(m["IoU"])
                        batch_iou = float(np.mean(batch_ious))
                # 单步损失标量（不除 accum_steps，用于显示）
                step_loss_item = float((loss_r + loss_b).item())
                window_loss_sum += step_loss_item
                window_count += 1
                steps_done += 1

                if window_count > 0:
                    avg_loss = window_loss_sum / window_count
                pbar.set_postfix_str(
                    f"avg_loss={avg_loss:.4f} loss={step_loss_item:.4f} IoU={batch_iou:.4f} "
                    f"Dr={dice_r_train:.4f} Fr={focal_r_train:.4f} "
                    f"Db={dice_b_train:.4f} Fb={focal_b_train:.4f} "
                    f"loss_r={loss_r_val:.4f} loss_b={loss_b_val:.4f}"
                )
                if step % PRINT_EVERY == 0 and window_count > 0:
                    window_loss_sum = 0.0
                    window_count = 0
                    accum_loss = 0.0

                # cleanup
                del imgs, gt_region, gt_boundary

            # for step, batch in pbar: 结束之后
            if steps_done > 0 and (steps_done % accum_steps != 0):
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # ===== region =====
            precision_r = train_tp_r / (train_tp_r + train_fp_r + 1e-8)
            recall_r = train_tp_r / (train_tp_r + train_fn_r + 1e-8)
            f1_r = 2 * precision_r * recall_r / (precision_r + recall_r + 1e-8)
            iou_r = train_tp_r / (train_tp_r + train_fp_r + train_fn_r + 1e-8)
            region_train = {
                "Precision": precision_r,
                "Recall": recall_r,
                "F1": f1_r,
                "IoU": iou_r
            }
            # ===== boundary =====
            precision_b = train_tp_b / (train_tp_b + train_fp_b + 1e-8)
            recall_b = train_tp_b / (train_tp_b + train_fn_b + 1e-8)
            f1_b = 2 * precision_b * recall_b / (precision_b + recall_b + 1e-8)
            iou_b = train_tp_b / (train_tp_b + train_fp_b + train_fn_b + 1e-8)
            boundary_train = {
                "Precision": precision_b,
                "Recall": recall_b,
                "F1": f1_b,
                "IoU": iou_b
            }
            train_loss = train_loss_sum / len(loader)
            # === 保存训练集指标 ===
            train_logs.append({
                "epoch": ep + 1,
                "train_loss": float(train_loss),
                "lr": optimizer.param_groups[0]["lr"],

                # region
                "region_thr": float(best_thr_region),
                "region_iou": float(region_train["IoU"]),
                "region_f1": float(region_train["F1"]),
                "region_precision": float(region_train["Precision"]),
                "region_recall": float(region_train["Recall"]),

                # boundary
                "boundary_thr": float(best_thr_boundary),
                "boundary_iou": float(boundary_train["IoU"]),
                "boundary_f1": float(boundary_train["F1"]),
                "boundary_precision": float(boundary_train["Precision"]),
                "boundary_recall": float(boundary_train["Recall"]),

                "global_step": int(global_step)
            })
            # end epoch: do validation
            sam.eval()
            val_loss = 0.0
            total_metrics = []
            val_dice_r, val_focal_r, val_loss_r = [], [], []
            val_dice_b, val_focal_b, val_loss_b = [], [], []
            all_pred_r, all_pred_b = [], []
            all_gt_r, all_gt_b = [], []

            with torch.no_grad():
                vpbar = tqdm(val_loader, desc=f"验证 Epoch {ep + 1}/{epochs}", ncols=100)
                for vbatch in vpbar:
                    imgs = torch.stack([b[0] for b in vbatch]).to(device).float()
                    gt_region = torch.stack([b[1] for b in vbatch]).to(device).long()
                    gt_boundary = torch.stack([b[2] for b in vbatch]).to(device).long()

                    mask_logits = torch.stack([b[3] for b in vbatch]).to(device).float()
                    pts_list_r = []
                    labels_list_r = []
                    boxes_list_r = []
                    pts_list_b = []
                    labels_list_b = []
                    boxes_list_b = []

                    for b in vbatch:
                        p_r = b[4]
                        l_r = b[5]
                        bx_r = b[6]
                        p_b = b[7]
                        l_b = b[8]
                        bx_b = b[9]
                        region_prompt = b[10]
                        boundary_prompt = b[11]

                        # squeeze 可能的 leading vbatch dim
                        # 如果形状是 (1, N, 2) -> squeeze -> (N,2)
                        if isinstance(p_r, torch.Tensor) and p_r.dim() == 3 and p_r.shape[0] == 1:
                            p_r = p_r.squeeze(0)
                        # 如果形状是 (N,2) 好的
                        if isinstance(l_r, torch.Tensor) and l_r.dim() == 2 and l_r.shape[0] == 1:
                            l_r = l_r.squeeze(0)
                        # boxes: 有时是 (1,4) 或 (4,)
                        if isinstance(bx_r, torch.Tensor) and bx_r.dim() == 2 and bx_r.shape[0] == 1:
                            bx_r = bx_r.squeeze(0)
                        # 如果是单个 box (4,) -> unsqueeze -> (1,4)
                        if isinstance(bx_r, torch.Tensor) and bx_r.dim() == 1:
                            bx_r = bx_r.unsqueeze(0)
                        pts_list_r.append(p_r.to(device))
                        labels_list_r.append(l_r.to(device))
                        boxes_list_r.append(bx_r.to(device))
                        # 如果形状是 (1, N, 2) -> squeeze -> (N,2)
                        if isinstance(p_b, torch.Tensor) and p_b.dim() == 3 and p_b.shape[0] == 1:
                            p_b = p_b.squeeze(0)
                        # 如果形状是 (N,2) 好的
                        if isinstance(l_b, torch.Tensor) and l_b.dim() == 2 and l_b.shape[0] == 1:
                            l_b = l_b.squeeze(0)
                        # boxes: 有时是 (1,4) 或 (4,)
                        if isinstance(bx_b, torch.Tensor) and bx_b.dim() == 2 and bx_b.shape[0] == 1:
                            bx_b = bx_b.squeeze(0)
                        # 如果是单个 box (4,) -> unsqueeze -> (1,4)
                        if isinstance(bx_b, torch.Tensor) and bx_b.dim() == 1:
                            bx_b = bx_b.unsqueeze(0)
                        pts_list_b.append(p_b.to(device))
                        labels_list_b.append(l_b.to(device))
                        boxes_list_b.append(bx_b.to(device))

                    try:
                        # 转换列表为批处理 tensor
                        pts_r_batch, labels_r_batch, boxes_r_batch = preprocess_prompts(
                            pts_list_r, labels_list_r, boxes_list_r, B=imgs.shape[0], device=device
                        )
                        pts_b_batch, labels_b_batch, boxes_b_batch = preprocess_prompts(
                            pts_list_b, labels_list_b, boxes_list_b, B=imgs.shape[0], device=device
                        )

                        pred_r, pred_b = forward_two_heads(
                            sam, imgs,
                            pts_r_batch, labels_r_batch, boxes_r_batch,
                            pts_b_batch, labels_b_batch, boxes_b_batch,
                            region_prompt, boundary_prompt
                        )

                    except Exception as e:
                        debug1_print(f"[ERROR val] forward_two_heads failed with error: {e}")
                        debug1_print("  pts_list_r shapes:", [getattr(x, 'shape', None) for x in pts_list_r])
                        debug1_print("  labels_list_r shapes:", [getattr(x, 'shape', None) for x in labels_list_r])
                        debug1_print("  boxes_list_r shapes:", [getattr(x, 'shape', None) for x in boxes_list_r])
                        debug1_print("  pts_list_b shapes:", [getattr(x, 'shape', None) for x in pts_list_b])
                        debug1_print("  labels_list_b shapes:", [getattr(x, 'shape', None) for x in labels_list_b])
                        debug1_print("  boxes_list_b shapes:", [getattr(x, 'shape', None) for x in boxes_list_b])
                        raise

                    pred_r_sig = torch.sigmoid(pred_r)
                    pred_b_sig = torch.sigmoid(pred_b)

                    # 🔑 上采样 ground truth 到预测分辨率
                    gt_region = F.interpolate(
                        gt_region.unsqueeze(1).float(),
                        size=pred_r_sig.shape[-2:],  # (H,W) of prediction
                        mode="nearest"
                    ).squeeze(1).long()

                    gt_boundary = F.interpolate(
                        gt_boundary.unsqueeze(1).float(),
                        size=pred_b_sig.shape[-2:],
                        mode="nearest"
                    ).squeeze(1).long()

                    # 计算损失（尽量把数学运算放在 autocast 内以节省显存）
                    dice_r_loss = dice(pred_r_sig, gt_region, reduction='mean')
                    focal_r_loss = focal(pred_r, gt_region, reduction='mean')

                    dice_b_loss = dice(pred_b_sig, gt_boundary, reduction='mean')
                    focal_b_loss = focal(pred_b, gt_boundary, reduction='mean')
                    dice_r_per_sample = dice(pred_r_sig, gt_region, reduction='none')  # [B]
                    focal_r_per_sample = focal(pred_r, gt_region, reduction='none')  # [B]

                    dice_b_per_sample = dice(pred_b_sig, gt_boundary, reduction='none')  # [B]
                    focal_b_per_sample = focal(pred_b, gt_boundary, reduction='none')  # [B]

                    if a == 0:
                        # 只用 FocalLoss
                        loss_r = focal_r_loss
                        loss_b = focal_b_loss
                    elif a == "mean":
                        # R 通道
                        norm_r_dice = dice_r_per_sample / (dice_r_per_sample.mean().detach() + 1e-8)
                        norm_r_focal = focal_r_per_sample / (focal_r_per_sample.mean().detach() + 1e-8)
                        loss_r = (norm_r_dice + norm_r_focal).mean()

                        # B 通道
                        norm_b_dice = dice_b_per_sample / (dice_b_per_sample.mean().detach() + 1e-8)
                        norm_b_focal = focal_b_per_sample / (focal_b_per_sample.mean().detach() + 1e-8)
                        loss_b = (norm_b_dice + norm_b_focal).mean()
                    elif a == "sum":
                        # 计算每个样本的scale
                        scale_r_per_sample = dice_r_per_sample.detach() + focal_r_per_sample.detach() + 1e-8
                        norm_r_dice = dice_r_per_sample / scale_r_per_sample
                        norm_r_focal = focal_r_per_sample / scale_r_per_sample
                        loss_r = (norm_r_dice + norm_r_focal).mean()

                        # Boundary 同理
                        scale_b_per_sample = dice_b_per_sample.detach() + focal_b_per_sample.detach() + 1e-8
                        norm_b_dice = dice_b_per_sample / scale_b_per_sample
                        norm_b_focal = focal_b_per_sample / scale_b_per_sample
                        loss_b = (norm_b_dice + norm_b_focal).mean()
                    elif a == "动态权重":
                        scale_r = dice_r_loss.detach() + focal_r_loss.detach() + 1e-8
                        weight_dice_r = focal_r_loss.detach() / scale_r
                        weight_focal_r = dice_r_loss.detach() / scale_r
                        loss_r = weight_dice_r * dice_r_loss + weight_focal_r * focal_r_loss

                        scale_b = dice_b_loss.detach() + focal_b_loss.detach() + 1e-8
                        weight_dice_b = focal_b_loss.detach() / scale_b
                        weight_focal_b = dice_b_loss.detach() / scale_b
                        loss_b = weight_dice_b * dice_b_loss + weight_focal_b * focal_b_loss
                    elif a == 0.4:
                        lambda_focal = 5.0
                        loss_r = a * dice_r_loss + (1 - a) * lambda_focal * focal_r_loss
                        loss_b = a * dice_b_loss + (1 - a) * lambda_focal * focal_b_loss

                    # 累积
                    val_dice_r.append(dice_r_loss.item())
                    val_focal_r.append(focal_r_loss.item())
                    val_loss_r.append(loss_r.item())
                    val_dice_b.append(dice_b_loss.item())
                    val_focal_b.append(focal_b_loss.item())
                    val_loss_b.append(loss_b.item())

                    if train_head == "region":
                        loss_v = loss_r.item()
                        val_loss += loss_v
                    elif train_head == "boundary":
                        loss_v = loss_b.item()
                        val_loss += loss_v
                    else:  # both
                        loss_v = (loss_r + loss_b).item()
                        val_loss += loss_v

                    all_pred_r.append(pred_r_sig.squeeze(1).cpu().numpy())  # (B,H,W)
                    all_pred_b.append(pred_b_sig.squeeze(1).cpu().numpy())
                    all_gt_r.append(gt_region.cpu().numpy())  # (B,H,W)
                    all_gt_b.append(gt_boundary.cpu().numpy())

            all_pred_r = np.concatenate(all_pred_r, axis=0)  # (N,H,W)
            all_pred_b = np.concatenate(all_pred_b, axis=0)
            all_gt_r = np.concatenate(all_gt_r, axis=0)
            all_gt_b = np.concatenate(all_gt_b, axis=0)

            val_loss /= max(1, len(val_loader))

            def scan_best_thresholds(all_pred_r, all_pred_b, all_gt_r, all_gt_b):
                # 🔹 验证集扫阈值
                best_thr_region, best_thr_boundary = 0.5, 0.5
                best_region_metrics = {"IoU": 0, "F1": 0, "Precision": 0, "Recall": 0}
                best_boundary_metrics = {"IoU": 0, "F1": 0, "Precision": 0, "Recall": 0}

                for thr in np.linspace(0.1, 0.9, 17):  # 每隔0.05扫
                    preds_r = (all_pred_r > thr).astype(np.uint8)
                    preds_b = (all_pred_b > thr).astype(np.uint8)

                    # 🔑 分别算 region 和 boundary 的指标
                    metrics_r = [compute_pixel_metrics(all_gt_r[i], preds_r[i]) for i in range(len(all_gt_r))]
                    metrics_b = [compute_pixel_metrics(all_gt_b[i], preds_b[i]) for i in range(len(all_gt_b))]

                    mean_r = {k: np.mean([m[k] for m in metrics_r]) for k in metrics_r[0].keys()}
                    mean_b = {k: np.mean([m[k] for m in metrics_b]) for k in metrics_b[0].keys()}
                    if train_head in ["region", "both"]:
                        if mean_r["IoU"] > best_region_metrics["IoU"]:
                            best_thr_region = thr
                            best_region_metrics = mean_r
                        pass  # 占位符，请保留原逻辑

                    # --- Boundary 扫描 ---
                    if train_head in ["boundary", "both"]:
                        if mean_b["F1"] > best_boundary_metrics["F1"]:
                            best_thr_boundary = thr
                            best_boundary_metrics = mean_b
                # 🔹 第二次精扫（在最优阈值附近 ±0.05）
                fine_range_r = np.linspace(best_thr_region - 0.05, best_thr_region + 0.05, 11)
                fine_range_r = np.clip(fine_range_r, 0.0, 1.0)  # 防止越界

                fine_range_b = np.linspace(best_thr_boundary - 0.05, best_thr_boundary + 0.05, 11)
                fine_range_b = np.clip(fine_range_b, 0.0, 1.0)

                # --- Region 扫描 ---
                if train_head in ["region", "both"]:
                    for thr in fine_range_r:
                        preds_r = (all_pred_r > thr).astype(np.uint8)
                        metrics_r = [compute_pixel_metrics(all_gt_r[i], preds_r[i]) for i in range(len(all_gt_r))]
                        mean_r = {k: np.mean([m[k] for m in metrics_r]) for k in metrics_r[0].keys()}
                        if mean_r["IoU"] > best_region_metrics["IoU"]:
                            best_thr_region, best_region_metrics = thr, mean_r
                    pass  # 占位符，请保留原逻辑

                # --- Boundary 扫描 ---
                if train_head in ["boundary", "both"]:
                    for thr in fine_range_b:
                        preds_b = (all_pred_b > thr).astype(np.uint8)
                        metrics_b = [compute_pixel_metrics(all_gt_b[i], preds_b[i]) for i in range(len(all_gt_b))]
                        mean_b = {k: np.mean([m[k] for m in metrics_b]) for k in metrics_b[0].keys()}
                        if mean_b["F1"] > best_boundary_metrics["F1"]:
                            best_thr_boundary, best_boundary_metrics = thr, mean_b
                    pass
                print(f"[验证] Region最优阈值={best_thr_region:.2f} | IoU={best_region_metrics['IoU']:.4f}, "
                      f"F1={best_region_metrics['F1']:.4f}, Precision={best_region_metrics['Precision']:.4f}, Recall={best_region_metrics['Recall']:.4f}")

                print(f"[验证] Boundary最优阈值={best_thr_boundary:.2f} | IoU={best_boundary_metrics['IoU']:.4f}, "
                      f"F1={best_boundary_metrics['F1']:.4f}, Precision={best_boundary_metrics['Precision']:.4f}, Recall={best_boundary_metrics['Recall']:.4f}")

                return best_thr_region, best_region_metrics, best_thr_boundary, best_boundary_metrics

            # 🔑 早停判断（用 loss）
            prev_best = early_stopping.best_score
            early_stopping(val_loss, sam)

            if early_stopping.best_score < prev_best:
                print("🟢 新的 best 模型出现，执行阈值扫描...")
                best_thr_region, best_region_metrics, best_thr_boundary, best_boundary_metrics = scan_best_thresholds(
                    all_pred_r, all_pred_b, all_gt_r, all_gt_b
                )
                torch.save({
                    "model": sam.state_dict(),
                    "best_thr_region": best_thr_region,
                    "best_thr_boundary": best_thr_boundary
                }, best_path)
            else:
                print("⚪ 本 epoch 未刷新 best，跳过阈值扫描。")

            # 🔑 早停判断（用 loss）
            if early_stopping.early_stop:
                break

            # 计算平均
            mean_dice_r = np.mean(val_dice_r)
            mean_focal_r = np.mean(val_focal_r)
            mean_loss_r = np.mean(val_loss_r)
            mean_dice_b = np.mean(val_dice_b)
            mean_focal_b = np.mean(val_focal_b)
            mean_loss_b = np.mean(val_loss_b)

            print(f"[VAL AVG] Dice_r={mean_dice_r:.4f}, Focal_r={mean_focal_r:.4f}, "
                  f"loss_r={mean_loss_r:.4f}, Dice_b={mean_dice_b:.4f}, "
                  f"Focal_b={mean_focal_b:.4f}, loss_b={mean_loss_b:.4f}")

            if device != "cpu":
                torch.cuda.empty_cache()
            # save checkpoint each epoch (store model + optimizer + scheduler + epoch)
            torch.save({
                "model": sam.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": ep + 1,
                "scaler": scaler.state_dict(),  # ← 必须保存
                "log_enc_strategy": enc_strategy,
                "global_step": global_step,
                "best_thr_region": best_thr_region,
                "best_thr_boundary": best_thr_boundary,
                # [新增] 3. 保存最佳指标，确保续传不报错
                "best_region_metrics": best_region_metrics,
                "best_boundary_metrics": best_boundary_metrics,
                "early_stopping": {
                    "best_score": early_stopping.best_score,
                    "counter": early_stopping.counter,
                    "early_stop": early_stopping.early_stop
                }
            }, ckpt_path)

            logs.append({
                "epoch": ep + 1,
                "val_loss": float(val_loss),

                # region 最优指标
                "region_thr": float(best_thr_region),
                "region_iou": float(best_region_metrics["IoU"]),
                "region_f1": float(best_region_metrics["F1"]),
                "region_precision": float(best_region_metrics["Precision"]),
                "region_recall": float(best_region_metrics["Recall"]),

                # boundary 最优指标
                "boundary_thr": float(best_thr_boundary),
                "boundary_iou": float(best_boundary_metrics["IoU"]),
                "boundary_f1": float(best_boundary_metrics["F1"]),
                "boundary_precision": float(best_boundary_metrics["Precision"]),
                "boundary_recall": float(best_boundary_metrics["Recall"]),

                "global_step": int(global_step)
            })

            print(f"[Epoch {ep + 1}/{epochs}] 验证损失: {val_loss:.6f}")

            sam.train()

        total_time = time.time() - total_start
        # save final model (Adapter adapters included in sam state_dict)
        torch.save(sam.state_dict(), os.path.join(CKPT_DIR, f"sam_adapter_final_{prompter_model}.pth"))
        # save logs
        # === 保存训练和验证日志到Excel ===
        df_val = pd.DataFrame(logs)
        df_train = pd.DataFrame(train_logs)

        df_val["total_time_sec"] = total_time
        df_train["total_time_sec"] = total_time

        excel_path = os.path.join(CKPT_DIR, "sam_adapter_log.xlsx")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_train.to_excel(writer, sheet_name="训练集结果", index=False)
            df_val.to_excel(writer, sheet_name="验证集结果", index=False)

        print("训练完成，总耗时 %.2f 秒；日志已保存。" % (total_time))

        if "region_iou" in df_val and "val_loss" in df_val:
            val = df_val["region_iou"].astype(float).values
            vls = df_val["val_loss"].astype(float).values

            K = min(5, len(val))
            tail = val[-K:]
            cov = float(np.std(tail) / (np.mean(tail) + 1e-12)) if K > 0 else 0.0

            best = float(np.max(val)) if len(val) > 0 else 0.0
            last = float(val[-1]) if len(val) > 0 else 0.0
            drawdown = best - last

            ma3_now = float(np.mean(val[-3:])) if len(val) >= 3 else last
            ma3_prev = float(np.mean(val[-6:-3])) if len(val) >= 6 else ma3_now
            slope = ma3_now - ma3_prev

            print(f"Val IoU CoV (last {K}): {cov:.4f}")
            print(f"Drawdown from best: {drawdown * 100:.2f} pp")
            print(f"MA(3) slope: {slope * 100:.2f} pp")

            flag_shake = cov > 0.01 or abs(slope) > 0.5 / 100
            flag_overfit = (drawdown > 0.5 / 100) and (len(vls) >= 3 and (vls[-1] > vls[-2] > vls[-3]))
            print("Shake?", flag_shake, "Overfit?", flag_overfit)
        if "region_iou" in df_val and "val_loss" in df_val:
            val = df_val["boundary_iou"].astype(float).values
            vls = df_val["val_loss"].astype(float).values

            K = min(5, len(val))
            tail = val[-K:]
            cov = float(np.std(tail) / (np.mean(tail) + 1e-12)) if K > 0 else 0.0

            best = float(np.max(val)) if len(val) > 0 else 0.0
            last = float(val[-1]) if len(val) > 0 else 0.0
            drawdown = best - last

            ma3_now = float(np.mean(val[-3:])) if len(val) >= 3 else last
            ma3_prev = float(np.mean(val[-6:-3])) if len(val) >= 6 else ma3_now
            slope = ma3_now - ma3_prev

            print(f"Val IoU CoV (last {K}): {cov:.4f}")
            print(f"Drawdown from best: {drawdown * 100:.2f} pp")
            print(f"MA(3) slope: {slope * 100:.2f} pp")

            flag_shake = cov > 0.01 or abs(slope) > 0.5 / 100
            flag_overfit = (drawdown > 0.5 / 100) and (len(vls) >= 3 and (vls[-1] > vls[-2] > vls[-3]))
            print("Shake?", flag_shake, "Overfit?", flag_overfit)

        # === 绘制训练 / 验证过程曲线 ===
        try:
            import matplotlib.pyplot as plt

            # ===== 验证曲线 =====
            epochs_val = [log["epoch"] for log in logs]
            val_losses = [log["val_loss"] for log in logs]
            val_region_iou = [log["region_iou"] for log in logs]
            val_boundary_iou = [log["boundary_iou"] for log in logs]

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs_val, val_losses, marker='o')
            plt.title("Validation Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(epochs_val, val_region_iou, label='Region IoU', marker='s')
            plt.plot(epochs_val, val_boundary_iou, label='Boundary IoU', marker='^')
            plt.title("Validation IoU per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("IoU")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            val_curve_path = os.path.join(CKPT_DIR, "val_curves.png")
            plt.savefig(val_curve_path, dpi=200)
            plt.close()

            # ===== 训练曲线 =====
            epochs_train = [log["epoch"] for log in train_logs]
            train_losses = [log["train_loss"] for log in train_logs]
            train_region_iou = [log["region_iou"] for log in train_logs]
            train_boundary_iou = [log["boundary_iou"] for log in train_logs]

            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs_train, train_losses, marker='o')
            plt.title("Train Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(epochs_train, train_region_iou, label='Region IoU', marker='s')
            plt.plot(epochs_train, train_boundary_iou, label='Boundary IoU', marker='^')
            plt.title("Train IoU per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("IoU")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            train_curve_path = os.path.join(CKPT_DIR, "train_curves.png")
            plt.savefig(train_curve_path, dpi=200)
            plt.close()

            print(f"✅ 验证曲线已保存: {val_curve_path}")
            print(f"✅ 训练曲线已保存: {train_curve_path}")

        except Exception as e:
            print(f"[WARN] 绘制训练曲线失败: {e}")
        return early_stopping.best_score, best_thr_region, best_thr_boundary

    # -----------------------
    # Test(region + boundary)
    # -----------------------
    def test_on_dataset(sam, dataset, best_thr_region, best_thr_boundary, out_dir, device):
        import traceback, sys
        sam.eval()
        os.makedirs(os.path.join(out_dir, "tif"), exist_ok=True)

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_out = os.path.join(out_dir, f"scores_{a}_{train_head}_{timestamp}.xlsx")

        df_exist = pd.DataFrame(columns=["图像"])
        if os.path.exists(excel_out):
            df_exist = pd.read_excel(excel_out, engine="openpyxl")
            if "图像" not in df_exist.columns:
                df_exist = pd.DataFrame(columns=["图像"])
            else:
                df_exist = df_exist[df_exist["图像"] != "总体平均"]

        all_score_rows = []
        total_start = time.time()

        dice = DiceLoss()
        focal = FocalLoss()

        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=safe_collate1)

        debug_once = False

        for i, batch in enumerate(tqdm(loader, total=len(dataset), desc="测试中")):
            debug_print(
                f"[DEBUG] batch {i} raw: type={type(batch)}, content types={[type(x) for x in batch] if isinstance(batch, (list, tuple)) else None}")

            if batch is None:
                debug_print(f"[WARN] batch {i} 全部无效，跳过", flush=True)
                raise

            # --- 打印 batch 的原始结构 ---
            debug_print(f"\n[test_on_dataset] batch index={i} 原始结构:", flush=True)
            for idx, item in enumerate(batch):
                try:
                    if isinstance(item, torch.Tensor):
                        debug_print(f"  element[{idx}] type={type(item)}, shape={tuple(item.shape)}", flush=True)
                    elif hasattr(item, '__len__') and not isinstance(item, str):
                        debug_print(f"  element[{idx}] type={type(item)}, len={len(item)}", flush=True)
                    else:
                        debug_print(f"  element[{idx}] type={type(item)}, value={item}", flush=True)
                except Exception as e:
                    debug1_print(f"  element[{idx}] 打印出错: {e}", flush=True)

            start_time = time.time()

            try:
                # 解包 batch
                img_t = batch[0]
                rg = batch[1]
                bd = batch[2]
                mask_prompt_logits = batch[3]
                pts_r = batch[4]
                labels_r = batch[5]
                boxes_r = batch[6]
                pts_b = batch[7]
                labels_b = batch[8]
                boxes_b = batch[9]
                nc_path = batch[12]
                region_prompt = batch[10]
                boundary_prompt = batch[11]

                if isinstance(nc_path, tuple):
                    nc_path = nc_path[0]  # 取出字符串
                debug_print(f"[test_on_dataset] nc_path type={type(nc_path)}, value={nc_path}")

                def _sh(x):
                    try:
                        return tuple(x.shape) if isinstance(x, torch.Tensor) else type(x)
                    except Exception:
                        return str(type(x))

                debug_print(f"[test_on_dataset] unpack shapes: img_t={_sh(img_t)}, rg={_sh(rg)}, bd={_sh(bd)}, "
                            f"mask_prompt_logits={_sh(mask_prompt_logits)}, "
                            f"pts_r={_sh(pts_r)}, labels_r={_sh(labels_r)}, boxes_r={_sh(boxes_r)}",
                            f"pts_b={_sh(pts_r)}, labels_b={_sh(labels_r)}, boxes_b={_sh(boxes_r)}",
                            flush=True)

                # === 规范 img_t ===
                if isinstance(img_t, torch.Tensor):
                    if img_t.dim() == 3:
                        debug_print("[test_on_dataset] img_t.dim()==3 -> unsqueeze(0)", flush=True)
                        img_t = img_t.unsqueeze(0)
                    elif img_t.dim() == 4:
                        debug_print("[test_on_dataset] img_t.dim()==4, 保持不变", flush=True)
                    else:
                        raise ValueError(f"Unexpected img_t.dim()={img_t.dim()}, shape={img_t.shape}")
                else:
                    img_t = torch.as_tensor(img_t)
                    if img_t.dim() == 3:
                        img_t = img_t.unsqueeze(0)

                img_t = img_t.to(device).float()
                debug_print(f"[test_on_dataset] normalized img_t.shape={tuple(img_t.shape)}", flush=True)
                debug_print("[test_on_dataset] img_t.dim()==", img_t.dim(), ", normalized img_t.shape=", img_t.shape)

                # --------- 修正 prompts 部分 ---------
                # === 处理 prompts ===
                # 注意：pts_r, labels_r, boxes_r 已经是 (1, N, 2) 或 (N, 2) 形状
                # 统一处理成列表形式供 preprocess_prompts 使用
                if pts_r.dim() == 3 and pts_r.shape[0] == 1:
                    pts_r = pts_r.squeeze(0)
                if labels_r.dim() == 2 and labels_r.shape[0] == 1:
                    labels_r = labels_r.squeeze(0)
                if boxes_r.dim() == 2 and boxes_r.shape[0] == 1:
                    boxes_r = boxes_r.squeeze(0)
                if boxes_r.dim() == 1:
                    boxes_r = boxes_r.unsqueeze(0)

                if pts_b.dim() == 3 and pts_b.shape[0] == 1:
                    pts_b = pts_b.squeeze(0)
                if labels_b.dim() == 2 and labels_b.shape[0] == 1:
                    labels_b = labels_b.squeeze(0)
                if boxes_b.dim() == 2 and boxes_b.shape[0] == 1:
                    boxes_b = boxes_b.squeeze(0)
                if boxes_b.dim() == 1:
                    boxes_b = boxes_b.unsqueeze(0)

                pts_list_r = [pts_r.to(device)]
                labels_list_r = [labels_r.to(device)]
                boxes_list_r = [boxes_r.to(device)]

                pts_list_b = [pts_b.to(device)]
                labels_list_b = [labels_b.to(device)]
                boxes_list_b = [boxes_b.to(device)]

                debug_print(
                    f"[test_on_dataset] AFTER reshape: pts_r shape={pts_list_r[0].shape}, "
                    f"labels_r shape={labels_list_r[0].shape}, boxes_r shape={boxes_list_r[0].shape}, "
                    f"pts_b shape={pts_list_b[0].shape}, labels_b shape={labels_list_b[0].shape}, boxes_b shape={boxes_list_b[0].shape}",
                    flush=True
                )

                # === forward ===
                try:
                    debug_print(
                        f"[test_on_dataset] about to call forward_two_heads: "
                        f"type={type(img_t)}, shape={img_t.shape}",
                        flush=True
                    )
                    # 使用 preprocess_prompts 统一处理点、框（可选，这里直接传列表）
                    # 注意：forward_two_heads 需要接收 pts, labels, boxes 的 batch 形式
                    # 由于 batch_size=1，我们可以直接传 unsqueezed 版本
                    pts_r_batch, labels_r_batch, boxes_r_batch = preprocess_prompts(
                        pts_list_r, labels_list_r, boxes_list_r, B=img_t.shape[0], device=device
                    )
                    pts_b_batch, labels_b_batch, boxes_b_batch = preprocess_prompts(
                        pts_list_b, labels_list_b, boxes_list_b, B=img_t.shape[0], device=device
                    )
                    outputs = forward_two_heads(
                        sam, img_t,
                        pts_r_batch, labels_r_batch, boxes_r_batch,
                        pts_b_batch, labels_b_batch, boxes_b_batch,
                        region_prompt, boundary_prompt
                    )

                except Exception as e:
                    debug1_print(f"[test_on_dataset] forward_two_heads failed on sample {i}, error={e}", flush=True)
                    traceback.print_exc(file=sys.stdout)
                    raise

                pred_r, pred_b = outputs
                pred_r_sig = torch.sigmoid(pred_r)
                pred_b_sig = torch.sigmoid(pred_b)

                gt_r = rg.squeeze(0).cpu().numpy()
                gt_b = bd.squeeze(0).cpu().numpy()

                gt_r_t = torch.from_numpy(gt_r).unsqueeze(0).unsqueeze(0).float().to(device)
                gt_b_t = torch.from_numpy(gt_b).unsqueeze(0).unsqueeze(0).float().to(device)

                # 计算损失（尽量把数学运算放在 autocast 内以节省显存）
                dice_r_loss = dice(pred_r_sig, gt_r_t, reduction='mean')
                focal_r_loss = focal(pred_r, gt_r_t.squeeze(1).long(), reduction='mean')

                dice_b_loss = dice(pred_b_sig, gt_b_t, reduction='mean')
                focal_b_loss = focal(pred_b, gt_b_t.squeeze(1).long(), reduction='mean')

                dice_r_per_sample = dice(pred_r_sig, gt_r_t, reduction='none')  # [B]
                focal_r_per_sample = focal(pred_r, gt_r_t.squeeze(1).long(), reduction='none')  # [B]

                dice_b_per_sample = dice(pred_b_sig, gt_b_t, reduction='none')  # [B]
                focal_b_per_sample = focal(pred_b, gt_b_t.squeeze(1).long(), reduction='none')  # [B]
                if a == 0:
                    # 只用 FocalLoss
                    loss_r = focal_r_loss
                    loss_b = focal_b_loss
                elif a == "mean":
                    # R 通道
                    norm_r_dice = dice_r_per_sample / (dice_r_per_sample.mean().detach() + 1e-8)
                    norm_r_focal = focal_r_per_sample / (focal_r_per_sample.mean().detach() + 1e-8)
                    loss_r = (norm_r_dice + norm_r_focal).mean()

                    # B 通道
                    norm_b_dice = dice_b_per_sample / (dice_b_per_sample.mean().detach() + 1e-8)
                    norm_b_focal = focal_b_per_sample / (focal_b_per_sample.mean().detach() + 1e-8)
                    loss_b = (norm_b_dice + norm_b_focal).mean()
                elif a == "sum":
                    # 计算每个样本的scale
                    scale_r_per_sample = dice_r_per_sample.detach() + focal_r_per_sample.detach() + 1e-8
                    norm_r_dice = dice_r_per_sample / scale_r_per_sample
                    norm_r_focal = focal_r_per_sample / scale_r_per_sample
                    loss_r = (norm_r_dice + norm_r_focal).mean()

                    # Boundary 同理
                    scale_b_per_sample = dice_b_per_sample.detach() + focal_b_per_sample.detach() + 1e-8
                    norm_b_dice = dice_b_per_sample / scale_b_per_sample
                    norm_b_focal = focal_b_per_sample / scale_b_per_sample
                    loss_b = (norm_b_dice + norm_b_focal).mean()
                elif a == "动态权重":
                    scale_r = dice_r_loss.detach() + focal_r_loss.detach() + 1e-8
                    weight_dice_r = focal_r_loss.detach() / scale_r
                    weight_focal_r = dice_r_loss.detach() / scale_r
                    loss_r = weight_dice_r * dice_r_loss + weight_focal_r * focal_r_loss

                    scale_b = dice_b_loss.detach() + focal_b_loss.detach() + 1e-8
                    weight_dice_b = focal_b_loss.detach() / scale_b
                    weight_focal_b = dice_b_loss.detach() / scale_b
                    loss_b = weight_dice_b * dice_b_loss + weight_focal_b * focal_b_loss
                elif a == 0.4:
                    lambda_focal = 5.0
                    loss_r = a * dice_r_loss + (1 - a) * lambda_focal * focal_r_loss
                    loss_b = a * dice_b_loss + (1 - a) * lambda_focal * focal_b_loss

                total_loss = (loss_r + loss_b).item()

                # ===== 新增：保存概率图 =====
                pred_r_prob = pred_r_sig[0, 0].detach().cpu().numpy()
                pred_b_prob = pred_b_sig[0, 0].detach().cpu().numpy()

                region_ref_path = dataset.rg[i]  # 第 i 个样本的 region 标签文件路径
                boundary_ref_path = dataset.bd[i]  # 第 i 个样本的 boundary 标签文件路径
                save_label_tif(
                    (pred_r_prob * 255).astype(np.uint8),
                    region_ref_path,
                    os.path.join(out_dir, "tif",
                                 os.path.basename(nc_path).replace(".nc", "_region_prob.tif"))
                )

                save_label_tif(
                    (pred_b_prob * 255).astype(np.uint8),
                    boundary_ref_path,
                    os.path.join(out_dir, "tif",
                                 os.path.basename(nc_path).replace(".nc", "_boundary_prob.tif"))
                )
                # === 后处理 ===
                pred_r_bin = (pred_r_sig[0, 0] > best_thr_region).cpu().numpy().astype(np.uint8)
                pred_b_bin = (pred_b_sig[0, 0] > best_thr_boundary).cpu().numpy().astype(np.uint8)

                save_label_tif(pred_r_bin, region_ref_path,
                               os.path.join(out_dir, "tif", os.path.basename(nc_path).replace(".nc", "_region.tif")))
                save_label_tif(pred_b_bin, boundary_ref_path,
                               os.path.join(out_dir, "tif", os.path.basename(nc_path).replace(".nc", "_boundary.tif")))

                if train_head == "region":
                    total_loss = loss_r.item()
                    region_metrics = compute_pixel_metrics(gt_r, pred_r_bin)
                    elapsed = time.time() - start_time
                    row = {
                        "图像": os.path.basename(nc_path),
                        "耗时(s)": elapsed,
                        "loss": total_loss,
                        **{f"region_{k}": v for k, v in region_metrics.items()},
                    }

                elif train_head == "boundary":
                    total_loss = loss_b.item()

                    boundary_metrics = compute_boundary_post_metrics(
                        pred_b_bin,
                        gt_b,
                        radius=BOUNDARY_RADIUS,
                        tolerance=BOUNDARY_TOLERANCE
                    )

                    elapsed = time.time() - start_time
                    row = {
                        "图像": os.path.basename(nc_path),
                        "耗时(s)": elapsed,
                        "loss": total_loss,
                        **{f"boundary_{k}": v for k, v in boundary_metrics.items()},
                    }

                else:  # both
                    total_loss = (loss_r + loss_b).item()
                    region_metrics = compute_pixel_metrics(gt_r, pred_r_bin)

                    boundary_metrics = compute_boundary_post_metrics(
                        pred_b_bin,
                        gt_b,
                        radius=BOUNDARY_RADIUS,
                        tolerance=BOUNDARY_TOLERANCE
                    )

                    elapsed = time.time() - start_time
                    row = {
                        "图像": os.path.basename(nc_path),
                        "耗时(s)": elapsed,
                        "loss": total_loss,
                        **{f"region_{k}": v for k, v in region_metrics.items()},
                        **{f"boundary_{k}": v for k, v in boundary_metrics.items()},
                    }
                all_score_rows.append(row)
                gc.collect()

            except Exception as e:
                debug1_print(
                    f"[ERROR] 样本 {i} 出错: {dataset.nc[i] if hasattr(dataset, 'nc') else 'unknown'}, 错误={e}",
                    flush=True)
                traceback.print_exc(file=sys.stdout)
                try:
                    debug_print(f"[DEBUG] 出错时 img_t.shape={getattr(img_t, 'shape', None)}, "
                                f"pts_r={getattr(pts_r, 'shape', None)}, labels_r={getattr(labels_r, 'shape', None)}, "
                                f"boxes_r={getattr(boxes_r, 'shape', None)}", flush=True)
                    debug_print(
                                f"pts_b={getattr(pts_b, 'shape', None)}, labels_b={getattr(labels_b, 'shape', None)}, "
                                f"boxes_b={getattr(boxes_b, 'shape', None)}", flush = True)
                except Exception:
                    pass

            if debug_once:
                debug_print("[test_on_dataset] debug_once=True, break after first batch", flush=True)
                break

        # === 合并 Excel ===
        df_new = pd.DataFrame(all_score_rows)
        df_all = pd.concat([df_exist, df_new], ignore_index=True)

        if not df_all.empty:
            df_all = df_all.drop_duplicates(subset=["图像"], keep="last")
            mean_metrics = df_all.mean(numeric_only=True)
            total_time = df_all["耗时(s)"].sum()
            avg_time = df_all.loc[df_all['图像'] != "总体平均", "耗时(s)"].fillna(0).mean()
            mean_overall_row = {"图像": "总体平均"}
            mean_overall_row.update(mean_metrics.to_dict())
            mean_overall_row["耗时(s)"] = total_time
            mean_overall_row["平均耗时(s)"] = avg_time
            df_all = pd.concat([df_all, pd.DataFrame([mean_overall_row])], ignore_index=True)

        with pd.ExcelWriter(excel_out, engine="openpyxl", mode='w') as writer:
            df_all.to_excel(writer, index=False)

        print("结果文件已保存为 Excel:", excel_out)
        print(f"测试总耗时: {time.time() - total_start:.2f} 秒")

    # -----------------------
    # Main: prepare datasets and run training + test
    # -----------------------
    # if __name__ == "__main__":
    # assert len(TRAIN_NC) == len(TRAIN_REGION), "训练集影像/region 数量不一致"
    # assert len(VAL_NC) == len(VAL_REGION), "验证集影像/region 数量不一致"

    if prompter_model.lower() == "deeplabv3p":
        from train_prompter_deeplabv3p import build_deeplabv3_plus
        def prompter_builder():
            return build_deeplabv3_plus(in_ch=3, num_classes=2)

    elif prompter_model.lower() == "unet":
        from train_prompter_U_net import build_UNet_plus
        def prompter_builder():
            return build_UNet_plus(in_ch=3, num_classes=2)

    if subset_ratio < 1.0:
        n_tr = int(len(TRAIN_NC) * subset_ratio)
        TRAIN_NC, TRAIN_REGION, TRAIN_BOUNDARY = TRAIN_NC[:n_tr], TRAIN_REGION[:n_tr], TRAIN_BOUNDARY[:n_tr]

        n_va = int(len(VAL_NC) * subset_ratio)
        VAL_NC, VAL_REGION, VAL_BOUNDARY = VAL_NC[:n_va], VAL_REGION[:n_va], VAL_BOUNDARY[:n_va]

    ds_tr = RSForSAMDataset(TRAIN_NC, TRAIN_REGION, TRAIN_BOUNDARY,
                            prompter_ckpt=PROMPTER_CKPT,
                            prompter_builder=prompter_builder,
                            bands=[0, 1, 2], crop=256)
    ds_va = RSForSAMDataset(VAL_NC, VAL_REGION, VAL_BOUNDARY,
                            prompter_ckpt=PROMPTER_CKPT,
                            prompter_builder=prompter_builder,
                            bands=[0, 1, 2], crop=256)
    '''
    val_loss, best_thr_region, best_thr_boundary = (
        train_sam_block_adapter(ds_tr, ds_va, sam_ckpt=SAM_CKPT,
                                epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                                device=DEVICE, prompter_builder=prompter_builder,
                                prompter_ckpt=PROMPTER_CKPT))
    '''
    # 加载最终模型
    best_state = os.path.join(CKPT_DIR, f"sam_adapter_best_{prompter_model}.pth")
    print(f"[INFO] Using checkpoint from: {best_state}")
    if not os.path.exists(best_state):
        best_state = os.path.join(CKPT_DIR, f"sam_adapter_final_{prompter_model}.pth")

    sam = build_sam_with_adapter(sam_type="vit_h", sam_ckpt=SAM_CKPT, device=DEVICE)
    ckpt = torch.load(best_state, map_location=DEVICE)
    if "model" in ckpt:
        sam.load_state_dict(ckpt["model"], strict=False)
    else:
        sam.load_state_dict(ckpt, strict=False)
    best_thr_region = ckpt.get('best_thr_region', 0.5)
    best_thr_boundary = ckpt.get('best_thr_boundary', 0.5)
    print(f"best_thr_region: {best_thr_region}")
    print(f"best_thr_boundary: {best_thr_boundary}")

    if do_test:
        test_nc, test_region, test_boundary = pair_by_stem(
            os.path.join(TEST_NC_DIR, "*.nc"),
            os.path.join(TEST_MASK_DIR, "*.tif"),
            os.path.join(TEST_BOUNDARY, "*.tif"),
        )
        ds_te = RSForSAMDataset(test_nc, test_region, test_boundary,
                                prompter_ckpt=PROMPTER_CKPT,
                                prompter_builder=prompter_builder,
                                bands=[0, 1, 2], crop=256)
        debug_print(len(ds_te), len(os.listdir(TEST_MASK_DIR)), len(os.listdir(TEST_BOUNDARY)))

        test_on_dataset(sam, ds_te, best_thr_region, best_thr_boundary, out_dir=OUT_DIR, device=DEVICE)

    return val_loss


'''  '''

# Adapter s_modettings
prompterel1 = "deeplabv3p"
prompter_model2 = "UNet"
adapter_dim = 128  # 缩放因子
LR = 5e-5  # 学习率
EPOCHS = 100
subset_ratio = 1
''' '''
a = 0.4
for train_head in ["boundary"]: #"boundary", "region"
    for enc_strategy in ["sandwich32"]:  # "sandwich6wen", "sandwich2", "sandwich6", "last2", "last4","sandwich"
        run_sam_with_adapter(prompter_model2, adapter_dim, LR, EPOCHS, subset_ratio, a, train_head, enc_strategy,
                             do_test=True)
