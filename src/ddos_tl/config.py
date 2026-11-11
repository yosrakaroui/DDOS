"""Chargement de la configuration YAML (avec héritage via la clé `_base`)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base", None)
    if base:
        cfg = _merge(load_config(p.parent / base), cfg)
    return cfg


def resolve(path_str: str | Path) -> Path:
    """Chemin relatif à la racine du projet -> chemin absolu."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p
