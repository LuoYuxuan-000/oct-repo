"""预处理后数据集加载器.

从 tools/prepare_octa500.py 生成的缓存目录读取:
  - splits.json: train/val/test 划分
  - samples/{id}.npz: 每个样本的 subarea 网格 + GT OSM
  - template_grid_uint8.npy: 模板极坐标网格
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

from pesnet.geometry.polar import PolarGridSpec, osm_to_normalized


def load_splits(processed_dir: Path) -> Dict[str, List[str]]:
    """读取 train/val/test 划分."""
    p = processed_dir / "splits.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_meta(processed_dir: Path) -> dict:
    """读取数据集元信息."""
    p = processed_dir / "meta.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_template_grid(processed_dir: Path) -> np.ndarray:
    """读取模板极坐标网格 (R,U,V) uint8."""
    return np.load(processed_dir / "template_grid_uint8.npy").astype(np.uint8, copy=False)


def load_sample_npz(processed_dir: Path, sample_id: str) -> Dict[str, np.ndarray]:
    """读取单个样本的 npz 缓存."""
    p = processed_dir / "samples" / f"{sample_id}.npz"
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class ProcessedSubareaIndex:
    """将 (sample_id, sub_idx) 展平为线性索引，驱动 DataLoader."""

    def __init__(self, processed_dir: Path, split: str) -> None:
        self.processed_dir = processed_dir
        splits = load_splits(processed_dir)
        self.sample_ids = list(splits[split])

        # 展平: 每个样本有多个 subarea
        items: List[Tuple[str, int]] = []
        for sid in self.sample_ids:
            d = load_sample_npz(processed_dir, sid)
            k = int(d["subarea_grids_uint8"].shape[0])
            items.extend((sid, j) for j in range(k))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.items[idx]


def build_torch_dataset(processed_dir: Path, split: str, spec: PolarGridSpec):
    """构建 PyTorch Dataset.

    每个样本返回:
      gs:   (1, R, U, V) subarea 极坐标网格
      gt:   (1, R, U, V) 模板极坐标网格（所有样本共享）
      y:    (1, U, V)    归一化 GT OSM [0,1]
      meta: dict 含 sample_id, sub_idx, center_xyz, phi
    """
    import torch
    from torch.utils.data import Dataset

    template_grid = torch.from_numpy(load_template_grid(processed_dir)).unsqueeze(0).float()
    index = ProcessedSubareaIndex(processed_dir, split)

    class _Dataset(Dataset):
        def __len__(self) -> int:
            return len(index)

        def __getitem__(self, i: int):
            sid, sub_idx = index[i]
            d = load_sample_npz(processed_dir, sid)

            # subarea 极坐标网格
            gs = torch.from_numpy(
                d["subarea_grids_uint8"][sub_idx].astype(np.float32)
            ).unsqueeze(0)  # (1, R, U, V)

            # 归一化 GT OSM
            gt_osm = d["gt_osm_uint8"].astype(np.uint8, copy=False)
            y = torch.from_numpy(osm_to_normalized(gt_osm, spec=spec)).unsqueeze(0)  # (1, U, V)

            meta: dict = {"sample_id": sid, "sub_idx": int(sub_idx)}
            if "center_xyz" in d:
                meta["center_xyz"] = torch.from_numpy(d["center_xyz"].astype(np.float32))
            if "phi" in d:
                meta["phi"] = torch.tensor(float(d["phi"]), dtype=torch.float32)

            return gs, template_grid, y, meta

    return _Dataset()
