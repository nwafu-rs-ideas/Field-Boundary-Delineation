# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d

# 全局调试开关
DEBUG = False  # 调试时设为 True，正常跑实验时设为 False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs, flush=True)

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        debug_print("[MASK DECODER] image_embeddings:", image_embeddings.shape)
        debug_print("[MASK DECODER] image_pe:", image_pe.shape)
        debug_print("[MASK DECODER] sparse_embeddings:",
              None if sparse_prompt_embeddings is None else sparse_prompt_embeddings.shape)
        debug_print("[MASK DECODER] dense_embeddings:",
              None if dense_prompt_embeddings is None else dense_prompt_embeddings.shape)


        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Batch and spatial sizes
        debug_print("[MASK DECODER] >>> entered forward")

        B, C, H, W = image_embeddings.shape
        debug_print(f"[PREDICT_MASKS] image_embeddings: {image_embeddings.shape}")

        # 1) prepare output tokens (iou + mask tokens)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)  # [num_mask_tokens, C]
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_mask_tokens, C]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  # [B, N_tokens, C]

        # 2) 处理 dense_prompt_embeddings
        if dense_prompt_embeddings is None:
            dense_prompt_embeddings = torch.zeros_like(image_embeddings)
            debug_print("[PREDICT_MASKS] dense_prompt_embeddings is None → zeros:", dense_prompt_embeddings.shape)
        else:
            debug_print("[PREDICT_MASKS] dense_prompt_embeddings in:", dense_prompt_embeddings.shape)

            if dense_prompt_embeddings.dim() == 4:
                # 批次对齐
                if dense_prompt_embeddings.shape[0] != B:
                    if dense_prompt_embeddings.shape[0] == 1:
                        dense_prompt_embeddings = dense_prompt_embeddings.expand(B, -1, -1, -1)
                        debug_print("[PREDICT_MASKS] expanded batch:", dense_prompt_embeddings.shape)
                    else:
                        raise ValueError(
                            f"dense_prompt_embeddings batch {dense_prompt_embeddings.shape[0]} "
                            f"!= image_embeddings batch {B}"
                        )

                # 空间对齐
                if dense_prompt_embeddings.shape[-2:] != (H, W):
                    debug_print("[PREDICT_MASKS] interpolate dense from", dense_prompt_embeddings.shape, "→", (H, W))
                    dense_prompt_embeddings = F.interpolate(
                        dense_prompt_embeddings,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                    debug_print("[PREDICT_MASKS] dense_prompt_embeddings after interp:", dense_prompt_embeddings.shape)
            else:
                raise ValueError(
                    f"Unexpected dense_prompt_embeddings shape {dense_prompt_embeddings.shape}, "
                    "expected [B,C,H,W]"
                )

        # 3) align image_pe
        debug_print("[PREDICT_MASKS] image_pe in:", image_pe.shape)
        if image_pe.shape[-2:] != (H, W):
            debug_print("[PREDICT_MASKS] interpolate image_pe from", image_pe.shape, "→", (H, W))
            image_pe = F.interpolate(image_pe, size=(H, W), mode="bilinear", align_corners=False)
        if image_pe.shape[0] != B:
            if image_pe.shape[0] == 1:
                image_pe = image_pe.expand(B, -1, -1, -1)
                debug_print("[PREDICT_MASKS] expanded image_pe batch:", image_pe.shape)
            else:
                raise ValueError(f"image_pe batch {image_pe.shape[0]} != {B}")

        # 4) compose inputs for transformer
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        debug_print("[PREDICT_MASKS] src:", src.shape, "pos_src:", pos_src.shape, "tokens:", tokens.shape)

        # 5) run transformer
        hs, src = self.transformer(src, pos_src, tokens)  # hs: [B, N_queries, C], src: [B, H*W, C]

        # 6) extract mask/iou tokens
        iou_token_out = hs[:, 0, :]  # [B, C]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]  # [B, num_mask_tokens, C]

        # 7) convert src back to [B, C, H, W] for upscaling
        b, n, c = src.shape  # n should equal H*W
        assert b == B, f"batch mismatch: transformer returned batch {b} vs image batch {B}"
        assert n == H * W, f"spatial mismatch: transformer n {n} != H*W {H * W}"
        src = src.transpose(1, 2).contiguous().view(b, c, H, W)  # [B, C, H, W]

        # 8) upscaling and hypernetwork
        upscaled_embedding = self.output_upscaling(src)  # [B, C_up, H_up, W_up]

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))  # [B, C_up]
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, num_mask_tokens, C_up]

        b, c_up, h_up, w_up = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c_up, h_up * w_up)).view(b, -1, h_up, w_up)

        # 9) iou preds
        iou_pred = self.iou_prediction_head(iou_token_out)  # [B, num_mask_tokens]

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
