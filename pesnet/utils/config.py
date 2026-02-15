"""YAML 配置文件加载与命令行覆盖."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class ConfigLoadResult:
    cfg: dict
    path: Path


def load_yaml(path: Path) -> ConfigLoadResult:
    """加载 YAML 配置文件."""
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"配置根节点必须为 dict，得到 {type(cfg)} ({p})")
    return ConfigLoadResult(cfg=cfg, path=p)


def deep_set(cfg: dict, key: str, value: Any) -> None:
    """设置嵌套键值: deep_set(cfg, "a.b.c", v) => cfg[a][b][c] = v."""
    parts = [p for p in str(key).split(".") if p]
    cur: Any = cfg
    for p in parts[:-1]:
        if p not in cur or cur[p] is None:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """应用 CLI 参数覆盖: --set training.lr=1e-4."""
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"无效覆盖 '{o}'，应为 key=value 格式")
        k, v_raw = o.split("=", 1)
        deep_set(cfg, k.strip(), yaml.safe_load(v_raw))
    return cfg
