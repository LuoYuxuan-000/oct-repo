"""PESNet 训练脚本.

用法:
  python train.py --config configs/pesnet_octa500.yaml
  python train.py --config configs/pesnet_octa500.yaml --set training.lr=5e-4
  python train.py --config configs/pesnet_octa500.yaml --resume outputs/runs/<run>/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml


def _seed_all(seed: int) -> None:
    """固定所有随机种子."""
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PESNet")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint 路径，用于断点续训")
    parser.add_argument("--run_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--set", type=str, action="append", default=[], help="覆盖配置项")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from pesnet.datasets.processed import build_torch_dataset, load_meta
    from pesnet.geometry.polar import PolarGridSpec, normalized_to_osm, osm_to_pointcloud
    from pesnet.losses.pesnet_loss import AlphaSchedule, PESNetLoss
    from pesnet.models.pesnet import PESNet, PESNetConfig
    from pesnet.utils.config import apply_overrides, load_yaml
    from pesnet.utils.metrics import chamfer_distance, density_error, ssim_2d

    # --- 加载配置 ---
    cfg = apply_overrides(load_yaml(Path(args.config)).cfg, args.set)
    seed = int(cfg.get("seed", 1337))
    _seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 数据 ---
    processed_dir = Path(cfg["data"]["processed_dir"])
    meta = load_meta(processed_dir)
    p = meta.get("polar") or cfg.get("polar") or {}
    spec = PolarGridSpec(R=int(p.get("R", 64)), U=int(p.get("U", 180)), V=int(p.get("V", 180)))

    train_ds = build_torch_dataset(processed_dir, "train", spec)
    val_ds = build_torch_dataset(processed_dir, "val", spec)

    tr_cfg = cfg.get("training", {})
    batch_size = int(tr_cfg.get("batch_size", 10))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_metrics_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # --- 模型 ---
    model = PESNet(PESNetConfig(
        R=spec.R, U=spec.U, V=spec.V,
        stride_uv=int(tr_cfg.get("stride_uv", 3)),
        upsample_uv=bool(tr_cfg.get("upsample_uv", True)),
    )).to(device)

    # --- 损失 + 优化器 ---
    epochs = int(tr_cfg.get("epochs", 300))
    lr = float(tr_cfg.get("lr", 1e-3))
    alpha_cfg = tr_cfg.get("alpha", {})
    alpha = AlphaSchedule(
        alpha0=float(alpha_cfg.get("alpha0", 1.0)),
        warmup_epochs=int(alpha_cfg.get("warmup_epochs", 50)),
        decay_to=float(alpha_cfg.get("decay_to", 0.0)),
    )
    perc_cfg = tr_cfg.get("perceptual", {})
    loss_fn = PESNetLoss(
        alpha=alpha,
        use_perceptual=bool(tr_cfg.get("use_perceptual", True)),
        perceptual_layer_names=perc_cfg.get("layers"),
        perceptual_layer_weights=perc_cfg.get("weights"),
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    use_amp = bool(tr_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- 输出目录 ---
    dataset_name = str(cfg.get("data", {}).get("dataset", "pesnet"))
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path("outputs/runs") / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{dataset_name}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # --- 断点续训 ---
    start_epoch, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"从 {args.resume} 恢复: epoch={start_epoch}, best_val={best_val:.6f}")

    # --- 验证指标计算 ---
    eval_every = int(tr_cfg.get("eval_every", 10))
    max_val_metrics = int(tr_cfg.get("max_val_metrics", 30))
    eval_cfg = cfg.get("eval", {})
    pc_mask_mode = str(eval_cfg.get("pointcloud_mask_mode", "paper"))
    bg_uv = eval_cfg.get("background_ref_uv", [5, 5])
    background_ref_uv = (int(bg_uv[0]), int(bg_uv[1]))

    def _val_metrics() -> tuple[float, float, float]:
        model.eval()
        ssim_vals, cd_vals, dens_vals = [], [], []
        with torch.no_grad():
            for j, (gs, gt, y, meta_b) in enumerate(val_metrics_loader):
                if j >= max_val_metrics:
                    break
                gs, gt, y = gs.to(device), gt.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(gs, gt).clamp(0.0, 1.0)
                pred_np = pred[0, 0, 0].cpu().numpy()
                gt_np = y[0, 0].cpu().numpy()
                ssim_vals.append(ssim_2d(pred_np, gt_np, data_range=1.0))

                if "center_xyz" in meta_b and "phi" in meta_b:
                    center = meta_b["center_xyz"][0].numpy()
                    phi = float(meta_b["phi"][0])
                    pc_pred = osm_to_pointcloud(
                        normalized_to_osm(pred_np, spec=spec), center, phi,
                        spec=spec, mask_mode=pc_mask_mode, background_ref_uv=background_ref_uv,
                    )
                    pc_gt = osm_to_pointcloud(
                        normalized_to_osm(gt_np, spec=spec), center, phi,
                        spec=spec, mask_mode=pc_mask_mode, background_ref_uv=background_ref_uv,
                    )
                    cd_vals.append(chamfer_distance(pc_pred, pc_gt))
                    dens_vals.append(density_error(pc_pred.shape[0], pc_gt.shape[0]))
        return (
            float(np.nanmean(ssim_vals)) if ssim_vals else float("nan"),
            float(np.nanmean(cd_vals)) if cd_vals else float("nan"),
            float(np.nanmean(dens_vals)) if dens_vals else float("nan"),
        )

    # --- 训练循环 ---
    history = {"train_loss": [], "val_loss": [], "val_ssim": [], "val_cd": [], "val_density": []}

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # 训练
        model.train()
        train_sum, train_n = 0.0, 0
        for gs, gt, y, _ in train_loader:
            gs, gt, y = gs.to(device), gt.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(gs, gt)
                loss = loss_fn(pred, y, epoch=epoch, total_epochs=epochs)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            train_sum += loss.item() * gs.size(0)
            train_n += gs.size(0)

        # 验证
        model.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for gs, gt, y, _ in val_loader:
                gs, gt, y = gs.to(device), gt.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(gs, gt)
                    loss = loss_fn(pred, y, epoch=epoch, total_epochs=epochs)
                val_sum += loss.item() * gs.size(0)
                val_n += gs.size(0)

        train_loss = train_sum / max(1, train_n)
        val_loss = val_sum / max(1, val_n)

        # 定期计算 SSIM/CD/Density
        val_ssim = val_cd = val_dens = float("nan")
        if eval_every > 0 and ((epoch + 1) % eval_every == 0 or (epoch + 1) == epochs):
            val_ssim, val_cd, val_dens = _val_metrics()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ssim"].append(val_ssim)
        history["val_cd"].append(val_cd)
        history["val_density"].append(val_dens)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        # 保存 checkpoint
        state = {"epoch": epoch, "model": model.state_dict(), "optim": optim.state_dict(),
                 "cfg": cfg, "best_val": best_val}
        if val_loss < best_val:
            best_val = val_loss
            state["best_val"] = best_val
            torch.save(state, ckpt_dir / "best.pt")
        torch.save(state, ckpt_dir / "last.pt")

        alpha_now = alpha.value(epoch, epochs)
        sec = time.time() - t0
        print(
            f"Epoch {epoch+1:04d}/{epochs} "
            f"train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f} "
            f"α={alpha_now:.3f} ssim={val_ssim:.4f} cd={val_cd:.4f} den={val_dens:.4f} "
            f"{sec:.1f}s"
        )

    print(f"训练完成. Best val: {best_val:.6f}")


if __name__ == "__main__":
    main()
