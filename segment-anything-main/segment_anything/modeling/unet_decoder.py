import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class AttentionGate(nn.Module):
    def __init__(self, ch_x, ch_skip):
        super().__init__()
        self.skip_align = nn.Conv2d(ch_skip, ch_x, 1)   # 通道对齐
        self.conv = nn.Sequential(
            nn.Conv2d(ch_x * 2, ch_x, 1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        # 空间尺寸对齐
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # 通道对齐
        skip = self.skip_align(skip)
        attn = self.conv(torch.cat([x, skip], dim=1))
        return x + attn * skip
class CrossAttentionSkip(nn.Module):
    def __init__(self, ch_x, ch_skip):
        super().__init__()
        self.skip_align = nn.Conv2d(ch_skip, ch_x, 1)   # 新增
        self.q = nn.Conv2d(ch_x, ch_x, 1)
        self.k = nn.Conv2d(ch_x, ch_x, 1)
        self.v = nn.Conv2d(ch_x, ch_x, 1)
        self.scale = ch_x ** -0.5

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear')
        skip = self.skip_align(skip)   # 对齐通道
        q = self.q(x)
        k = self.k(skip)
        v = self.v(skip)
        B, C, H, W = q.shape
        q = q.view(B, C, -1)
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)
        attn = torch.softmax((q.transpose(1,2) @ k) * self.scale, dim=-1)
        out = (attn @ v.transpose(1,2)).transpose(1,2).view(B, C, H, W)
        return x + out
# ================== BiFPN ==================
class BiFPN(nn.Module):
    def __init__(self, ch=256):
        super().__init__()

        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))
        self.w3 = nn.Parameter(torch.ones(2))

        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, f1, f2, f3, f4):
        # f1=128, f2=64, f3=32, f4=16

        # top-down
        w = torch.relu(self.w1)
        P3 = (w[0]*f3 + w[1]*F.interpolate(f4, size=f3.shape[-2:], mode='bilinear')) / (w.sum()+1e-6)
        P3 = self.conv3(P3)

        w = torch.relu(self.w2)
        P2 = (w[0]*f2 + w[1]*F.interpolate(P3, size=f2.shape[-2:], mode='bilinear')) / (w.sum()+1e-6)
        P2 = self.conv2(P2)

        w = torch.relu(self.w3)
        P1 = (w[0]*f1 + w[1]*F.interpolate(P2, size=f1.shape[-2:], mode='bilinear')) / (w.sum()+1e-6)
        P1 = self.conv1(P1)

        return P1, P2, P3

# ================== Decoder ==================
class UNetDecoder(nn.Module):
    def __init__(self, in_ch=256):
        super().__init__()

        self.bifpn = BiFPN(in_ch)

        # attention gates
        self.attn1 = AttentionGate(128, 256)  # x 通道 128, skip(P3) 通道 256 self.attn1 = Attention(128, 256)
        self.attn2 = AttentionGate(64, 256)  # x 通道 64,  skip(P2) 通道 256
        self.attn3 = AttentionGate(64, 256)  # x 通道 64,  skip(P1) 通道 256

        # upsampling
        self.up1 = nn.ConvTranspose2d(in_ch, 128, 2, stride=2)
        self.conv1 = DoubleConv(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(64, 64)

        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv3 = DoubleConv(64, 64)

    def forward(self, x, features):
        f1, f2, f3, f4 = features  # 128,64,32,16

        # ===== BiFPN =====
        P1, P2, P3 = self.bifpn(f1, f2, f3, f4)

        # ===== stage1 ===== 32→64
        x = self.up1(x)
        x = self.attn1(x, P3)
        x = self.conv1(x)

        # ===== stage2 ===== 64→128
        x = self.up2(x)
        x = self.attn2(x, P2)
        x = self.conv2(x)

        # ===== stage3 ===== 128→256
        x = self.up3(x)
        x = self.attn3(x, P1)
        x = self.conv3(x)

        return x