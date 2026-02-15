"""评估指标: Chamfer Distance / SSIM / Density.

论文使用:
  - CD: 0.5 * (mean_a2b + mean_b2a) 的 L2 距离（非平方）
  - SSIM: 2D OSM 层面的结构相似性
  - Density: |1 - N_pred / N_gt|，越低表示有效区域越完整
"""

from __future__ import annotations

import numpy as np


def chamfer_distance(
    a_xyz: np.ndarray,
    b_xyz: np.ndarray,
    *,
    max_points: int = 20_000,
) -> float:
    """对称 Chamfer Distance (L2, 非平方).

    CD = 0.5 * (mean_{a} min_{b} ||a-b|| + mean_{b} min_{a} ||b-a||)
    """
    a = np.asarray(a_xyz, dtype=np.float32)
    b = np.asarray(b_xyz, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return float("inf")

    # 随机下采样防止内存溢出
    rng = np.random.default_rng(0)
    if a.shape[0] > max_points:
        a = a[rng.choice(a.shape[0], size=max_points, replace=False)]
    if b.shape[0] > max_points:
        b = b[rng.choice(b.shape[0], size=max_points, replace=False)]

    from scipy.spatial import cKDTree
    ta, tb = cKDTree(a), cKDTree(b)
    da, _ = tb.query(a, k=1)  # a->b 最近距离
    db, _ = ta.query(b, k=1)  # b->a 最近距离
    return 0.5 * float(np.mean(da) + np.mean(db))


def density_error(n_pred: int, n_gt: int) -> float:
    """点云密度误差: |1 - N_pred/N_gt|."""
    if n_gt <= 0:
        return float("inf")
    return float(abs(1.0 - (float(n_pred) / float(n_gt))))


def ssim_2d(pred: np.ndarray, gt: np.ndarray, *, data_range: float = 1.0) -> float:
    """2D SSIM，用于评估 OSM 回归质量."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    try:
        from skimage.metrics import structural_similarity
        ssim_val, _ = structural_similarity(gt, pred, data_range=float(data_range), full=True)
        return float(ssim_val)
    except Exception:
        return float("nan")
