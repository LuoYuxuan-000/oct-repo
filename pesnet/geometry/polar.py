"""极坐标变换与 Polar Voxelization Block (PVB).

实现论文 Zhang et al. (MICCAI 2023) Eq.(2)-(4):
  r = sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2)
  u = arctan((z-cz) / sqrt((x-cx)^2+(y-cy)^2)) + pi/2   (elevation, [0,pi))
  v = arctan((y-cy) / (x-cx)) + pi/2                     (azimuth,  [0,pi))

体素化 (PVB):
  G[round(r/phi), round(u*U/pi), round(v*V/pi)] = 1

OSM (Ocular Surface Map):
  OSM(u,v) = max_r G(r,u,v)    取最外表面
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


@dataclass(frozen=True)
class PolarGridSpec:
    """极坐标网格参数（论文默认: R=64, U=180, V=180）."""

    R: int = 64
    U: int = 180
    V: int = 180


def _as_f32(x: np.ndarray) -> np.ndarray:
    """确保输入为 float32."""
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


# ---------------------------------------------------------------------------
# 笛卡尔 <-> 极坐标  (论文 Eq.2)
# ---------------------------------------------------------------------------

def cartesian_to_polar(
    points_xyz: np.ndarray,
    center_xyz: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """笛卡尔坐标 -> 极坐标 (r, u, v).

    Args:
        points_xyz: (N,3) 点云
        center_xyz: (3,) 极坐标原点（论文取 top en-face 中心）
    Returns:
        r: 径向距离
        u: elevation 角 [0, pi)
        v: azimuth 角 [0, pi)
    """
    points_xyz = _as_f32(points_xyz).reshape(-1, 3)
    center_xyz = _as_f32(center_xyz).reshape(1, 3)

    d = points_xyz - center_xyz
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

    r = np.sqrt(dx * dx + dy * dy + dz * dz)

    # u: elevation — arctan(dz / sqrt(dx^2+dy^2)) + pi/2
    xy = np.sqrt(dx * dx + dy * dy)
    u = np.arctan2(dz, np.maximum(xy, eps)) + (np.pi / 2.0)

    # v: azimuth — arctan(dy/dx) + pi/2
    # 注意：使用 arctan 而非 arctan2，匹配论文的半平面约定
    small_dx = np.abs(dx) <= eps
    ratio = np.divide(dy, dx, out=np.zeros_like(dy), where=~small_dx)
    ratio = np.where(
        small_dx,
        np.where(dy > 0, np.inf, np.where(dy < 0, -np.inf, 0.0)),
        ratio,
    )
    v = np.arctan(ratio) + (np.pi / 2.0)

    # 截断到 [0, pi)
    pi_eps = np.float32(np.pi - 1e-6)
    u = np.clip(u, 0.0, pi_eps)
    v = np.clip(v, 0.0, pi_eps)
    return r.astype(np.float32), u.astype(np.float32), v.astype(np.float32)


def polar_to_cartesian(
    r: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    center_xyz: np.ndarray,
) -> np.ndarray:
    """极坐标 -> 笛卡尔坐标（cartesian_to_polar 的逆变换）.

    令 theta = u - pi/2, phi_angle = v - pi/2:
      dz  = r * sin(theta)
      rho = r * cos(theta)
      dx  = rho * cos(phi_angle)
      dy  = rho * sin(phi_angle)
    """
    r64 = np.asarray(r, dtype=np.float64).reshape(-1)
    u64 = np.asarray(u, dtype=np.float64).reshape(-1)
    v64 = np.asarray(v, dtype=np.float64).reshape(-1)
    if not (r64.shape == u64.shape == v64.shape):
        raise ValueError(f"r/u/v 形状必须一致，得到 {r64.shape}, {u64.shape}, {v64.shape}")

    center = np.asarray(center_xyz, dtype=np.float64).reshape(1, 3)
    theta = u64 - (np.pi / 2.0)
    phi_angle = v64 - (np.pi / 2.0)

    dz = r64 * np.sin(theta)
    rho = r64 * np.cos(theta)
    dx = rho * np.cos(phi_angle)
    dy = rho * np.sin(phi_angle)

    xyz = np.stack([dx, dy, dz], axis=-1) + center
    return xyz.astype(np.float32)


# ---------------------------------------------------------------------------
# 离散化 binning  (论文 Eq.3)
# ---------------------------------------------------------------------------

def _bin_polar(
    r: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    phi: float,
    spec: PolarGridSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """连续极坐标 -> 离散网格索引.

    r_idx = round(r / phi),  u_idx = round(u * U / pi),  v_idx = round(v * V / pi)
    """
    if phi <= 0:
        raise ValueError(f"phi 必须 > 0，得到 {phi}")

    ridx = np.rint(r / float(phi)).astype(np.int32)
    uidx = np.rint(u * (float(spec.U) / np.pi)).astype(np.int32)
    vidx = np.rint(v * (float(spec.V) / np.pi)).astype(np.int32)

    ridx = np.clip(ridx, 0, int(spec.R) - 1)
    uidx = np.clip(uidx, 0, int(spec.U) - 1)
    vidx = np.clip(vidx, 0, int(spec.V) - 1)
    return ridx, uidx, vidx


# ---------------------------------------------------------------------------
# PVB 体素化  (论文 Sec 2.2)
# ---------------------------------------------------------------------------

def voxelize_to_grid(
    points_xyz: np.ndarray,
    center_xyz: np.ndarray,
    phi: float,
    *,
    spec: PolarGridSpec = PolarGridSpec(),
) -> np.ndarray:
    """Polar Voxelization Block (PVB): 点云 -> 二值占据网格 (R,U,V).

    对每个点执行: G[round(r/phi), round(u*U/pi), round(v*V/pi)] = 1
    """
    pts = _as_f32(points_xyz).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((spec.R, spec.U, spec.V), dtype=np.uint8)

    r, u, v = cartesian_to_polar(pts, center_xyz)
    ridx, uidx, vidx = _bin_polar(r, u, v, phi=float(phi), spec=spec)

    grid = np.zeros((spec.R, spec.U, spec.V), dtype=np.uint8)
    grid[ridx, uidx, vidx] = 1
    return grid


# ---------------------------------------------------------------------------
# OSM 生成与转换  (论文 Eq.4)
# ---------------------------------------------------------------------------

def grid_to_osm(grid_ruv: np.ndarray) -> np.ndarray:
    """(R,U,V) 占据网格 -> OSM (U,V)，每个 (u,v) 取 max r 索引."""
    g = np.asarray(grid_ruv)
    if g.ndim != 3:
        raise ValueError(f"grid 必须为 (R,U,V)，得到 {g.shape}")

    osm = np.zeros((g.shape[1], g.shape[2]), dtype=np.uint8)
    ridx, uidx, vidx = np.where(g != 0)
    if ridx.size == 0:
        return osm
    np.maximum.at(osm, (uidx, vidx), ridx.astype(np.uint8))
    return osm


def points_to_osm(
    points_xyz: np.ndarray,
    center_xyz: np.ndarray,
    phi: float,
    *,
    spec: PolarGridSpec = PolarGridSpec(),
) -> np.ndarray:
    """点云 -> OSM (U,V)，直接取最外表面 (max r_bin)."""
    pts = _as_f32(points_xyz).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((spec.U, spec.V), dtype=np.uint8)

    r, u, v = cartesian_to_polar(pts, center_xyz)
    ridx, uidx, vidx = _bin_polar(r, u, v, phi=float(phi), spec=spec)
    osm = np.zeros((spec.U, spec.V), dtype=np.uint8)
    np.maximum.at(osm, (uidx, vidx), ridx.astype(np.uint8))
    return osm


# ---------------------------------------------------------------------------
# OSM -> 点云 逆变换（推理阶段使用）
# ---------------------------------------------------------------------------

MaskMode = Literal["paper", "nonzero", "all"]


def osm_to_pointcloud(
    osm_uv: np.ndarray,
    center_xyz: np.ndarray,
    phi: float,
    *,
    spec: PolarGridSpec = PolarGridSpec(),
    mask_mode: MaskMode = "paper",
    background_ref_uv: Tuple[int, int] = (5, 5),
    float_eps: float = 1e-3,
) -> np.ndarray:
    """OSM (U,V) -> 点云 (N,3)，通过逆极坐标变换.

    论文去噪策略:
      - "paper": 仅转换 OSM(u,v) != OSM(5,5) 的像素
      - "nonzero": 仅转换非零像素
      - "all": 转换所有像素

    支持 uint8 OSM [0,63] 和归一化 float OSM [0,1] 两种输入。
    """
    if phi <= 0:
        raise ValueError(f"phi 必须 > 0，得到 {phi}")

    # 兼容 torch.Tensor
    try:
        import torch
        if isinstance(osm_uv, torch.Tensor):
            osm = osm_uv.detach().cpu().numpy()
        else:
            osm = np.asarray(osm_uv)
    except Exception:
        osm = np.asarray(osm_uv)

    if osm.ndim != 2 or osm.shape != (spec.U, spec.V):
        raise ValueError(f"osm 形状 {osm.shape} 需为 ({spec.U},{spec.V})")

    # 判断是否为归一化 float [0,1]，若是则转换回 r_bin
    is_float = np.issubdtype(osm.dtype, np.floating)
    is_norm = bool(is_float and float(osm.max(initial=0.0)) <= 1.0 + 1e-6)

    if is_norm:
        osm01 = osm.astype(np.float32, copy=False)
        ridx_f = np.clip(np.rint(osm01 * float(spec.R - 1)), 0, spec.R - 1).astype(np.float32)
    else:
        osm01 = None
        ridx_f = osm.astype(np.float32, copy=False)

    # 构建前景掩码
    if mask_mode == "all":
        mask = np.ones((spec.U, spec.V), dtype=bool)
    elif mask_mode == "nonzero":
        if is_norm and osm01 is not None:
            mask = np.abs(osm01) > float_eps
        else:
            mask = ridx_f != 0.0
    elif mask_mode == "paper":
        bg_u, bg_v = background_ref_uv
        if is_norm and osm01 is not None:
            bg_val = float(osm01[bg_u, bg_v])
            mask = np.abs(osm01 - bg_val) > float_eps
        else:
            bg_val = float(ridx_f[bg_u, bg_v])
            mask = ridx_f != bg_val
    else:
        raise ValueError(f"未知 mask_mode: {mask_mode}")

    u_idx, v_idx = np.where(mask)
    if u_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    r = ridx_f[u_idx, v_idx] * float(phi)
    u = (u_idx.astype(np.float32) * np.pi) / float(spec.U)
    v = (v_idx.astype(np.float32) * np.pi) / float(spec.V)
    return polar_to_cartesian(r, u, v, center_xyz)


# ---------------------------------------------------------------------------
# OSM 归一化（训练用）
# ---------------------------------------------------------------------------

def osm_to_normalized(
    osm_uint8: np.ndarray,
    *,
    spec: PolarGridSpec = PolarGridSpec(),
) -> np.ndarray:
    """GT OSM [0, R-1] -> 归一化 [0, 1]，用于网络训练."""
    return np.asarray(osm_uint8, dtype=np.uint8).astype(np.float32) / float(spec.R - 1)


def normalized_to_osm(
    norm_uv: np.ndarray,
    *,
    spec: PolarGridSpec = PolarGridSpec(),
) -> np.ndarray:
    """归一化 [0,1] -> uint8 r_bin [0, R-1]，用于推理后转换."""
    norm = np.asarray(norm_uv, dtype=np.float32)
    return np.clip(np.rint(norm * float(spec.R - 1)), 0, spec.R - 1).astype(np.uint8)
