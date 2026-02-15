"""PESNet 损失函数.

论文 Loss = α · L1_smooth + L_perc

- L1_smooth: SmoothL1Loss (Huber loss, beta=1.0)
- L_perc:    VGG16 感知损失 (Johnson et al., ECCV 2016)
- α 调度:    前 warmup_epochs 个 epoch 固定为 alpha0，之后线性衰减到 decay_to
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# α 衰减策略
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlphaSchedule:
    """论文 α 衰减: 前 warmup_epochs 固定，之后线性衰减."""

    alpha0: float = 1.0
    warmup_epochs: int = 50
    decay_to: float = 0.0

    def value(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 0:
            return float(self.alpha0)
        if epoch < self.warmup_epochs:
            return float(self.alpha0)
        if total_epochs <= self.warmup_epochs:
            return float(self.decay_to)
        frac = (epoch - self.warmup_epochs) / float(max(1, total_epochs - self.warmup_epochs - 1))
        frac = float(max(0.0, min(1.0, frac)))
        return float(self.alpha0 + frac * (self.decay_to - self.alpha0))


# ---------------------------------------------------------------------------
# VGG16 感知损失
# ---------------------------------------------------------------------------

class VGG16PerceptualLoss(nn.Module):
    """基于 VGG16 特征图的感知损失 (Johnson et al., ECCV 2016).

    论文使用的层: relu1_2, relu2_2, relu3_3
    """

    _NAME_TO_IDX = {
        "relu1_1": 1, "relu1_2": 3,
        "relu2_1": 6, "relu2_2": 8,
        "relu3_1": 11, "relu3_2": 13, "relu3_3": 15,
        "relu4_1": 18, "relu4_2": 20, "relu4_3": 22,
        "relu5_1": 25, "relu5_2": 27, "relu5_3": 29,
    }

    def __init__(
        self,
        *,
        layer_names: Sequence[str] = ("relu1_2", "relu2_2", "relu3_3"),
        layer_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()

        # 解析层索引
        self.layer_indices = sorted(self._NAME_TO_IDX[n] for n in layer_names)
        self.layer_weights = (
            list(layer_weights) if layer_weights is not None
            else [1.0] * len(self.layer_indices)
        )

        # 加载预训练 VGG16 并冻结参数
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = vgg.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

        # ImageNet 归一化参数
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """将单通道 OSM 转为 VGG 可接受的 3 通道 224×224 输入."""
        # 接受 (B,1,H,W) 或 (B,1,1,H,W)
        if x.dim() == 5 and x.size(2) == 1:
            x = x.squeeze(2)
        x = x.repeat(1, 3, 1, 1)  # 单通道复制为 3 通道
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """提取指定层的特征图."""
        feats: List[torch.Tensor] = []
        cur = x
        for i, layer in enumerate(self.features):
            cur = layer(cur)
            if i in self.layer_indices:
                feats.append(cur)
            if i >= self.layer_indices[-1]:
                break
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fp = self._extract(self._prep(pred))
        ft = self._extract(self._prep(target))
        loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        for a, b, w in zip(fp, ft, self.layer_weights):
            loss = loss + w * F.mse_loss(a, b, reduction="mean")
        return loss


# ---------------------------------------------------------------------------
# PESNet 组合损失
# ---------------------------------------------------------------------------

class PESNetLoss(nn.Module):
    """Loss = α · SmoothL1(pred, gt) + PerceptualLoss(pred, gt).

    α 按 AlphaSchedule 衰减，使训练后期更侧重感知损失。
    """

    def __init__(
        self,
        *,
        alpha: AlphaSchedule = AlphaSchedule(),
        use_perceptual: bool = True,
        perceptual_layer_names: Sequence[str] | None = None,
        perceptual_layer_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.use_perceptual = bool(use_perceptual)
        self.perc = (
            VGG16PerceptualLoss(
                layer_names=perceptual_layer_names or ("relu1_2", "relu2_2", "relu3_3"),
                layer_weights=perceptual_layer_weights,
            )
            if self.use_perceptual else None
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        # pred 可能是 (B,1,1,H,W)，target 是 (B,1,H,W)
        if pred.dim() == 5 and target.dim() == 4 and pred.size(2) == 1:
            pred_4d = pred.squeeze(2)
        else:
            pred_4d = pred

        l1 = F.smooth_l1_loss(pred_4d, target, beta=1.0, reduction="mean")

        if not self.use_perceptual:
            return l1

        lp = self.perc(pred, target)
        a = self.alpha.value(epoch, total_epochs)
        return a * l1 + lp
