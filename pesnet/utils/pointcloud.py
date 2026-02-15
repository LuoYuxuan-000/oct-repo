"""点云后处理: 连通分量滤波 + 统计离群点去除."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CCFilterStats:
    """连通分量滤波统计."""
    n_in: int
    n_out: int
    num_components: int
    num_kept: int


@dataclass(frozen=True)
class SORFilterStats:
    """统计离群点去除统计."""
    n_in: int
    n_out: int
    k: int
    threshold: float


def filter_connected_components(
    points_xyz: np.ndarray,
    *,
    center_xy: np.ndarray,
    radius: float,
    min_cluster_size: int = 50,
    min_rho_mean: float = 0.0,
) -> tuple[np.ndarray, CCFilterStats]:
    """连通分量滤波: 去除太小或离极坐标原点太近的噪声簇.

    PESNet 常见伪影: 背景像素重建出原点附近的竖直"柱子"，此函数可过滤。
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    n_in = pts.shape[0]
    if n_in == 0:
        return pts, CCFilterStats(0, 0, 0, 0)

    center_xy = np.asarray(center_xy, dtype=np.float32).reshape(2)
    tree = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, r=float(radius))

    visited = np.zeros(n_in, dtype=bool)
    keep = np.zeros(n_in, dtype=bool)
    n_comp, n_kept = 0, 0

    for i in range(n_in):
        if visited[i]:
            continue
        # BFS 找连通分量
        stack, comp = [i], []
        visited[i] = True
        while stack:
            idx = stack.pop()
            comp.append(idx)
            for j in neighbors[idx]:
                if not visited[j]:
                    visited[j] = True
                    stack.append(j)
        n_comp += 1
        comp_idx = np.array(comp, dtype=np.int32)

        # 过滤: 太小的簇
        if len(comp) < min_cluster_size:
            continue
        # 过滤: XY 平面离原点太近的簇（柱子伪影）
        if min_rho_mean > 0:
            rho = np.linalg.norm(pts[comp_idx, :2] - center_xy, axis=1)
            if float(np.mean(rho)) < min_rho_mean:
                continue

        keep[comp_idx] = True
        n_kept += 1

    return pts[keep], CCFilterStats(n_in, int(keep.sum()), n_comp, n_kept)


def filter_statistical_outliers(
    points_xyz: np.ndarray,
    *,
    k: int = 16,
    std_ratio: float = 3.0,
) -> tuple[np.ndarray, SORFilterStats]:
    """统计离群点去除 (SOR): 基于 kNN 平均距离的鲁棒阈值过滤."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    n_in = pts.shape[0]
    if n_in <= k:
        return pts, SORFilterStats(n_in, n_in, k, float("inf"))

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    mean_d = np.mean(dists[:, 1:], axis=1).astype(np.float32)

    med = float(np.median(mean_d))
    mad = float(np.median(np.abs(mean_d - med))) + 1e-8
    thr = med + std_ratio * 1.4826 * mad  # 鲁棒阈值

    mask = mean_d <= thr
    return pts[mask], SORFilterStats(n_in, int(mask.sum()), k, thr)
