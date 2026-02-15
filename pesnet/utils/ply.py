"""PLY 点云文件保存，用于 MeshLab/CloudCompare 可视化."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_pointcloud_ply(path: str | Path, xyz: np.ndarray) -> None:
    """保存 (N,3) 点云为 ASCII PLY 格式."""
    p = Path(path)
    pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)

    header = "\n".join([
        "ply",
        "format ascii 1.0",
        f"element vertex {pts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ])

    with p.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for x, y, z in pts:
            f.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
