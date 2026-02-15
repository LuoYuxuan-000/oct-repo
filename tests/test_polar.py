"""极坐标变换的往返测试."""

import numpy as np

from pesnet.geometry.polar import (
    PolarGridSpec, cartesian_to_polar, polar_to_cartesian,
    voxelize_to_grid, grid_to_osm, osm_to_pointcloud,
)


def test_polar_roundtrip():
    """cartesian -> polar -> cartesian 往返精度."""
    rng = np.random.default_rng(0)
    center = np.array([10.0, 20.0, 0.0], dtype=np.float32)

    r = rng.uniform(0.0, 200.0, size=10_000).astype(np.float32)
    u = rng.uniform(0.0, np.pi, size=10_000).astype(np.float32)
    v = rng.uniform(0.0, np.pi, size=10_000).astype(np.float32)

    pts = polar_to_cartesian(r, u, v, center)
    r2, u2, v2 = cartesian_to_polar(pts, center)

    assert np.max(np.abs(r2 - r)) < 5e-4
    assert np.max(np.abs(u2 - np.clip(u, 0.0, np.pi - 1e-6))) < 5e-4
    assert np.max(np.abs(v2 - np.clip(v, 0.0, np.pi - 1e-6))) < 5e-4


def test_osm_voxelize_roundtrip():
    """OSM -> 点云 -> 体素化 -> OSM 往返一致性."""
    spec = PolarGridSpec(R=64, U=180, V=180)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    phi = 5.0

    rng = np.random.default_rng(1)
    osm = np.zeros((spec.U, spec.V), dtype=np.uint8)
    u_idx = rng.integers(0, spec.U, size=500)
    v_idx = rng.integers(0, spec.V, size=500)
    osm[u_idx, v_idx] = rng.integers(1, spec.R, size=500).astype(np.uint8)

    pts = osm_to_pointcloud(osm, center, phi, spec=spec, mask_mode="paper")
    grid = voxelize_to_grid(pts, center, phi, spec=spec)
    osm2 = grid_to_osm(grid)

    mask = osm != osm[5, 5]
    assert np.array_equal(osm2[mask], osm[mask])
