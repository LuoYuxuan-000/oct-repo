"""OCTA-500 数据预处理: 生成训练缓存 + 模板.

从 OCTA-500 原始数据生成:
  - splits.json: train/val/test 划分 (70%/20%/10%)
  - samples/{id}.npz: 每个 volume 的 GT OSM + 5 个 subarea 网格 + phi 等
  - template_grid_uint8.npy: 从 top-7 训练样本平均 + 高斯平滑的模板

用法:
  python tools/prepare_octa500.py \\
    --octa500_root ../data/OCTA-500 \\
    --out_root data/processed/octa500_usage1 \\
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import scipy.io
import yaml


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _load_surface(mat_path: Path, *, layer_key: str = "Layer", surface_index: int = -1) -> np.ndarray:
    """从 GT_Layers .mat 文件提取 RPE 层表面 (H,W)."""
    mat = scipy.io.loadmat(mat_path)
    layer = np.asarray(mat[layer_key])  # (K, H, W)
    idx = surface_index if surface_index >= 0 else layer.shape[0] + surface_index
    return layer[idx].astype(np.float32)


def _surface_to_pointcloud(surface: np.ndarray, *, stride: int = 2, x_off: int = 0, y_off: int = 0) -> np.ndarray:
    """2D 层表面 (H,W) -> 3D 点云 (N,3): x=col, y=row, z=depth."""
    h, w = surface.shape
    xs = np.arange(0, w, stride, dtype=np.float32) + x_off
    ys = np.arange(0, h, stride, dtype=np.float32) + y_off
    xx, yy = np.meshgrid(xs, ys)
    depth = surface[::stride, ::stride].astype(np.float32)
    return np.stack([xx.ravel(), yy.ravel(), depth.ravel()], axis=1)


def _compute_phi(points: np.ndarray, center: np.ndarray, R: int) -> float:
    """根据 r 的 99 分位数自动估计 phi."""
    from pesnet.geometry.polar import cartesian_to_polar
    r, _, _ = cartesian_to_polar(points, center)
    return float(np.quantile(r, 0.99)) / float(max(1, R - 1))


def _subarea_boxes(h: int, w: int, size: float) -> list[tuple[int, int, int, int]]:
    """生成 5 个确定性 subarea 框: center + 四象限."""
    win_w = int(round(w * size)) if size <= 1 else int(round(size))
    win_h = int(round(h * size)) if size <= 1 else int(round(size))
    win_w, win_h = max(8, min(win_w, w)), max(8, min(win_h, h))

    def _box(cx: float, cy: float):
        x0 = max(0, min(int(round(cx - win_w / 2)), w - win_w))
        y0 = max(0, min(int(round(cy - win_h / 2)), h - win_h))
        return (x0, x0 + win_w, y0, y0 + win_h)

    return [_box(w * cx, h * cy) for cx, cy in [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OCTA-500 for PESNet")
    parser.add_argument("--octa500_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--subarea_size", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--template_k", type=int, default=7)
    parser.add_argument("--split_seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    _setup_logging(args.verbose)
    from pesnet.geometry.polar import PolarGridSpec, points_to_osm, voxelize_to_grid, osm_to_pointcloud
    from scipy.ndimage import gaussian_filter

    octa_root = Path(args.octa500_root).resolve()
    out_root = Path(args.out_root).resolve()
    samples_dir = out_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    spec = PolarGridSpec(R=64, U=180, V=180)
    layers_dir = octa_root / "Label" / "GT_Layers"
    oct_dir = octa_root / "OCTA_6mm" / "OCT"

    # 获取有效 ID（同时有 GT_Layers 和 OCT 数据的）
    layer_ids = {p.stem for p in layers_dir.glob("*.mat")}
    oct_ids = sorted(p.name for p in oct_dir.iterdir() if p.is_dir()) if oct_dir.exists() else []
    ids = sorted(set(oct_ids) & layer_ids)

    # 划分数据集
    rng = np.random.default_rng(args.split_seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train, n_val = int(n * 0.7), int(n * 0.2)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    (out_root / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # 逐 volume 处理
    template_candidates = []

    for idx, sid in enumerate(ids):
        surface = _load_surface(layers_dir / f"{sid}.mat")
        h, w = surface.shape
        center = np.array([w / 2.0, h / 2.0, 0.0], dtype=np.float32)
        full_pts = _surface_to_pointcloud(surface, stride=args.stride)
        phi = _compute_phi(full_pts, center, spec.R)

        # GT OSM
        gt_osm = points_to_osm(full_pts, center, phi, spec=spec)

        # 5 个 subarea 网格
        boxes = _subarea_boxes(h, w, args.subarea_size)
        sub_grids = []
        for x0, x1, y0, y1 in boxes:
            sub_pts = _surface_to_pointcloud(surface[y0:y1, x0:x1], stride=args.stride, x_off=x0, y_off=y0)
            sub_grids.append(voxelize_to_grid(sub_pts, center, phi, spec=spec))

        np.savez_compressed(samples_dir / f"{sid}.npz",
            gt_osm_uint8=gt_osm.astype(np.uint8),
            subarea_grids_uint8=np.stack(sub_grids).astype(np.uint8),
            center_xyz=center, phi=np.float32(phi),
        )

        if sid in train_ids:
            template_candidates.append((sid, gt_osm, phi))

        if (idx + 1) % 25 == 0 or (idx + 1) == n:
            print(f"Processed {idx+1}/{n}")

    # 构建模板: top-k 训练样本平均 + 高斯平滑
    k = min(args.template_k, len(template_candidates))
    selected = template_candidates[:k]
    mean_osm = np.mean([osm.astype(np.float32) for _, osm, _ in selected], axis=0)
    smooth_osm = gaussian_filter(mean_osm, sigma=1.5)
    template_osm = np.clip(np.rint(smooth_osm), 0, spec.R - 1).astype(np.uint8)
    np.save(out_root / "template_osm_uint8.npy", template_osm)

    phi_template = float(np.median([p for _, _, p in selected]))
    center_ref = np.array([surface.shape[1] / 2.0, surface.shape[0] / 2.0, 0.0], dtype=np.float32)
    template_pc = osm_to_pointcloud(template_osm, center_ref, phi_template, spec=spec, mask_mode="all")
    template_grid = voxelize_to_grid(template_pc, center_ref, phi_template, spec=spec)
    np.save(out_root / "template_grid_uint8.npy", template_grid)

    meta = {
        "polar": {"R": spec.R, "U": spec.U, "V": spec.V},
        "splits": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
    }
    (out_root / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    print(f"完成. train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")


if __name__ == "__main__":
    main()
