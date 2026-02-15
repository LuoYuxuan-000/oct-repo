"""PESNet 双分支网络架构 (论文 Fig.1).

S-branch (Shape regression):  局部 subarea 极坐标网格输入
A-branch (Anatomical prior): 模板极坐标网格输入
两分支通过 RFB (R-wise Fusion Block) 逐层融合，最终输出 OSM ∈ [0,1].

输入:  (B, 1, R=64, U=180, V=180) × 2
输出:  (B, 1, 1, U=180, V=180)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PESNetConfig:
    """PESNet 超参数，与论文对齐."""

    R: int = 64
    U: int = 180
    V: int = 180
    in_channels: int = 1

    # RB1~RB4 各向异性卷积 stride (论文: stride=(2,3,3))
    stride_uv: int = 3
    # 各向异性卷积后上采样保持 U,V 不变
    upsample_uv: bool = True

    debug_shapes: bool = False


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------

class _ConvBNAct(nn.Module):
    """3D 卷积 + BatchNorm + 激活函数."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        act: str | None = "relu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm3d(out_ch)
        if act is None:
            self.act: nn.Module = nn.Identity()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"未知激活函数: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock3D(nn.Module):
    """ResNet 风格的 3D 残差块 (论文 Fig.1(c))."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = _ConvBNAct(in_ch, out_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1), act="relu")
        self.conv2 = _ConvBNAct(out_ch, out_ch, kernel_size=(3, 3, 3), padding=(1, 1, 1), act=None)
        self.shortcut = (
            _ConvBNAct(in_ch, out_ch, kernel_size=(1, 1, 1), act=None)
            if in_ch != out_ch else nn.Identity()
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.conv2(self.conv1(x)) + self.shortcut(x))


class RB(nn.Module):
    """Residual Block (RB) — 论文中的基本处理单元.

    - RB1~RB4 (mode="reduce"): 残差块 + 各向异性卷积 (2,3,3) 降低 R 维
    - RB5 (mode="pool"): 残差块 + MaxPool (4,1,1) 将 R 降到 1
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        mode: str,
        stride_uv: int,
        upsample_uv: bool,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.upsample_uv = bool(upsample_uv)
        self.stride_uv = int(stride_uv)
        self.res = ResidualBlock3D(in_ch, out_ch)

        if mode == "reduce":
            # 各向异性卷积: kernel=(2,3,3), stride=(2,stride_uv,stride_uv)
            self.reduce = _ConvBNAct(
                out_ch, out_ch,
                kernel_size=(2, 3, 3),
                stride=(2, self.stride_uv, self.stride_uv),
                padding=(0, 1, 1),
                act="relu",
            )
        elif mode == "pool":
            self.pool = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))
        else:
            raise ValueError(f"未知 RB mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.res(x)
        if self.mode == "reduce":
            _, _, _, u0, v0 = y.shape
            y = self.reduce(y)
            # 上采样恢复 U,V 尺寸
            if self.upsample_uv and (y.shape[-2] != u0 or y.shape[-1] != v0):
                y = F.interpolate(y, size=(y.shape[-3], u0, v0), mode="trilinear", align_corners=False)
            return y
        return self.pool(y)


# ---------------------------------------------------------------------------
# R-wise Fusion Block (RFB) — 论文 Fig.1(c), Sec 2.2
# ---------------------------------------------------------------------------

def _interleave_r_slices(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """沿 R 维交错拼接: (B,C,R,U,V) × 2 -> (B,C,2R,U,V).

    RFB1-4 的核心操作：将 S-branch 和 A-branch 的特征按 R 维交替排列，
    使得相同深度层的特征相邻，方便后续卷积学习跨分支相关性。
    """
    b, c, r, u, v = s.shape
    stacked = torch.stack([s, a], dim=3)  # (B,C,R,2,U,V)
    return stacked.reshape(b, c, 2 * r, u, v)


def _interleave_channels(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """沿通道维交错拼接: (B,C,1,U,V) × 2 -> (B,2C,1,U,V).

    RFB5 的核心操作：R=1 时改为通道维交错，使 Sigmoid 输出融合两分支信息。
    """
    b, c, _r, u, v = s.shape
    s2 = s.squeeze(2)
    a2 = a.squeeze(2)
    stacked2 = torch.stack([s2, a2], dim=2)  # (B,C,2,U,V)
    interleaved = stacked2.permute(0, 2, 1, 3, 4).reshape(b, 2 * c, u, v)
    return interleaved.unsqueeze(2)  # (B,2C,1,U,V)


class RFB(nn.Module):
    """R-wise Fusion Block.

    - RFB1-4 (kind="rfb1_4"): R维交错 + conv(2,3,3) stride(2,1,1) + BN + ReLU
    - RFB5   (kind="rfb5"):   通道交错 + conv(1,3,3) + BN + Sigmoid -> OSM [0,1]
    """

    def __init__(self, channels: int, *, kind: str) -> None:
        super().__init__()
        self.kind = kind

        if kind == "rfb1_4":
            self.fuse = _ConvBNAct(
                channels, channels,
                kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1),
                act="relu",
            )
        elif kind == "rfb5":
            self.fuse = _ConvBNAct(
                2 * channels, 1,
                kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                act="sigmoid",
            )
        else:
            raise ValueError(f"未知 RFB kind: {kind}")

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self.kind == "rfb1_4":
            return self.fuse(_interleave_r_slices(s, a))
        return self.fuse(_interleave_channels(s, a))


# ---------------------------------------------------------------------------
# PESNet 完整网络 — 论文 Fig.1(b)
# ---------------------------------------------------------------------------

class PESNet(nn.Module):
    """PESNet 双分支网络.

    论文核心架构:
      S-branch: gs -> RB1 -> RB2 -> RB3 -> RB4 -> RB5 (局部形状回归)
      A-branch: gt -> RB1 -> RB2 -> RB3 -> RB4 -> RB5 (解剖先验)
      每层通过 RFB 融合 A-branch 特征到 S-branch

    通道变化: 1 -> 16 -> 32 -> 64 -> 128 -> 128 -> 32 -> 1
    R维变化:  64 -> 32 -> 16 -> 8 -> 4 -> 1 -> 1

    输入: gs, gt 均为 (B, 1, R, U, V)
    输出: OSM (B, 1, 1, U, V)，值域 [0,1]
    """

    def __init__(self, cfg: PESNetConfig = PESNetConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        su, uu = cfg.stride_uv, cfg.upsample_uv

        # S-branch（权重不共享）
        self.s_rb1 = RB(cfg.in_channels, 16, mode="reduce", stride_uv=su, upsample_uv=uu)
        self.s_rb2 = RB(16, 32, mode="reduce", stride_uv=su, upsample_uv=uu)
        self.s_rb3 = RB(32, 64, mode="reduce", stride_uv=su, upsample_uv=uu)
        self.s_rb4 = RB(64, 128, mode="reduce", stride_uv=su, upsample_uv=uu)
        self.s_rb5 = RB(128, 128, mode="pool", stride_uv=su, upsample_uv=uu)

        # A-branch（权重不共享）
        self.a_rb1 = RB(cfg.in_channels, 16, mode="reduce", stride_uv=su, upsample_uv=uu)
        self.a_rb2 = RB(16, 32, mode="reduce", stride_uv=su, upsample_uv=uu)
        self.a_rb3 = RB(32, 64, mode="reduce", stride_uv=su, upsample_uv=uu)
        self.a_rb4 = RB(64, 128, mode="reduce", stride_uv=su, upsample_uv=uu)
        self.a_rb5 = RB(128, 128, mode="pool", stride_uv=su, upsample_uv=uu)

        # RFB1-4: 逐层融合 A->S
        self.rfb1 = RFB(16, kind="rfb1_4")
        self.rfb2 = RFB(32, kind="rfb1_4")
        self.rfb3 = RFB(64, kind="rfb1_4")
        self.rfb4 = RFB(128, kind="rfb1_4")

        # RB5 之后: 1×1×1 降通道 128->32
        self.s_reduce = _ConvBNAct(128, 32, kernel_size=(1, 1, 1), act="relu")
        self.a_reduce = _ConvBNAct(128, 32, kernel_size=(1, 1, 1), act="relu")

        # RFB5: 通道融合 + Sigmoid 输出
        self.rfb5 = RFB(32, kind="rfb5")

    def forward(
        self,
        gs: torch.Tensor,
        gt: torch.Tensor,
        *,
        return_shapes: bool = False,
    ):
        shapes: Dict[str, Tuple[int, ...]] = {}

        def _rec(name: str, x: torch.Tensor) -> None:
            if self.cfg.debug_shapes or return_shapes:
                shapes[name] = tuple(x.shape)

        _rec("in_gs", gs)
        _rec("in_gt", gt)

        # 逐层: RB 提取特征 + RFB 融合
        s1 = self.s_rb1(gs);   a1 = self.a_rb1(gt)
        s1f = self.rfb1(s1, a1);  _rec("s1f", s1f)

        s2 = self.s_rb2(s1f);  a2 = self.a_rb2(a1)
        s2f = self.rfb2(s2, a2);  _rec("s2f", s2f)

        s3 = self.s_rb3(s2f);  a3 = self.a_rb3(a2)
        s3f = self.rfb3(s3, a3);  _rec("s3f", s3f)

        s4 = self.s_rb4(s3f);  a4 = self.a_rb4(a3)
        s4f = self.rfb4(s4, a4);  _rec("s4f", s4f)

        s5 = self.s_rb5(s4f);  a5 = self.a_rb5(a4)
        _rec("s5", s5); _rec("a5", a5)

        # 降通道 + RFB5 输出
        s5r = self.s_reduce(s5)
        a5r = self.a_reduce(a5)
        out = self.rfb5(s5r, a5r)  # (B, 1, 1, U, V)
        _rec("out", out)

        if return_shapes:
            return out, shapes
        return out
