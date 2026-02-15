"""PESNet 评估脚本.

用法:
  python eval.py --config configs/pesnet_octa500.yaml
  python eval.py --config configs/pesnet_octa500.yaml --run_dir outputs/runs/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PESNet (CD / SSIM / Density)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="最多评估多少个样本")
    parser.add_argument("--set", type=str, action="append", default=[])
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from pesnet.datasets.processed import build_torch_dataset, load_meta
    from pesnet.geometry.polar import PolarGridSpec, osm_to_pointcloud
    from pesnet.models.pesnet import PESNet, PESNetConfig
    from pesnet.utils.config import apply_overrides, load_yaml
    from pesnet.utils.metrics import chamfer_distance, density_error, ssim_2d
    from pesnet.utils.ply import save_pointcloud_ply
    from pesnet.utils.pointcloud import filter_connected_components, filter_statistical_outliers

    # --- 配置 ---
    cfg = apply_overrides(load_yaml(Path(args.config)).cfg, args.set)
    processed_dir = Path(cfg["data"]["processed_dir"])
    meta = load_meta(processed_dir)
    p = meta.get("polar") or cfg.get("polar") or {}
    spec = PolarGridSpec(R=int(p.get("R", 64)), U=int(p.get("U", 180)), V=int(p.get("V", 180)))

    # --- 加载模型 ---
    run_dir = Path(args.run_dir) if args.run_dir else Path("outputs/runs/last")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (run_dir / "checkpoints" / "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    tr_cfg = cfg.get("training", {})
    model = PESNet(PESNetConfig(
        R=spec.R, U=spec.U, V=spec.V,
        stride_uv=int(tr_cfg.get("stride_uv", 3)),
        upsample_uv=bool(tr_cfg.get("upsample_uv", True)),
    ))
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- 数据 ---
    test_ds = build_torch_dataset(processed_dir, "test", spec)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    eval_cfg = cfg.get("eval", {})
    bg_uv = eval_cfg.get("background_ref_uv", [5, 5])
    background_ref_uv = (int(bg_uv[0]), int(bg_uv[1]))

    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 评估循环 ---
    cd_vals, ssim_vals, dens_vals = [], [], []

    for i, (gs, gt, y, meta_b) in enumerate(test_loader):
        if args.limit and len(ssim_vals) >= args.limit:
            break

        sid = meta_b["sample_id"][0] if "sample_id" in meta_b else f"item{i:04d}"
        sub_idx = int(meta_b["sub_idx"][0]) if "sub_idx" in meta_b else 0

        gs, gt, y = gs.to(device), gt.to(device), y.to(device)
        with torch.no_grad():
            pred = model(gs, gt).clamp(0.0, 1.0)

        pred_np = pred[0, 0, 0].cpu().numpy().astype(np.float32)
        gt_np = y[0, 0].cpu().numpy().astype(np.float32)

        ssim = ssim_2d(pred_np, gt_np, data_range=1.0)
        ssim_vals.append(ssim)

        if "center_xyz" in meta_b and "phi" in meta_b:
            center = meta_b["center_xyz"][0].numpy().astype(np.float32)
            phi = float(meta_b["phi"][0])

            pc_pred = osm_to_pointcloud(
                pred_np, center, phi, spec=spec,
                mask_mode="paper", background_ref_uv=background_ref_uv,
            )
            pc_gt = osm_to_pointcloud(
                gt_np, center, phi, spec=spec,
                mask_mode="paper", background_ref_uv=background_ref_uv,
            )

            # 点云后处理
            if pc_pred.size:
                pc_pred, _ = filter_connected_components(
                    pc_pred, center_xy=center[:2],
                    radius=2.0 * phi, min_cluster_size=50, min_rho_mean=3.0 * phi,
                )
                pc_pred, _ = filter_statistical_outliers(pc_pred, k=16, std_ratio=3.0)

            cd_vals.append(chamfer_distance(pc_pred, pc_gt))
            dens_vals.append(density_error(pc_pred.shape[0], pc_gt.shape[0]))
        else:
            cd_vals.append(float("nan"))
            dens_vals.append(float("nan"))
            pc_pred = np.zeros((0, 3), dtype=np.float32)
            pc_gt = np.zeros((0, 3), dtype=np.float32)

        stem = f"{sid}_sub{sub_idx}"
        np.save(out_dir / f"{stem}_pred_osm.npy", pred_np)
        np.save(out_dir / f"{stem}_gt_osm.npy", gt_np)
        if pc_pred.size:
            save_pointcloud_ply(out_dir / f"{stem}_pred.ply", pc_pred)
        if pc_gt.size:
            save_pointcloud_ply(out_dir / f"{stem}_gt.ply", pc_gt)

        print(
            f"[{len(ssim_vals):04d}/{len(test_ds)}] {stem} "
            f"ssim={ssim:.4f} cd={cd_vals[-1]:.4f} dens={dens_vals[-1]:.4f}"
        )

    # --- 汇总结果 ---
    summary = {
        "checkpoint": str(ckpt_path),
        "num_items": len(ssim_vals),
        "cd_mean": float(np.nanmean(cd_vals)),
        "cd_std": float(np.nanstd(cd_vals)),
        "ssim_mean": float(np.nanmean(ssim_vals)),
        "ssim_std": float(np.nanstd(ssim_vals)),
        "density_mean": float(np.nanmean(dens_vals)),
        "density_std": float(np.nanstd(dens_vals)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
