"""Lecture et nettoyage des CSV CIC (CICFlowMeter)."""
from __future__ import annotations

from pathlib import Path
import ctypes
import gc

import numpy as np
import pandas as pd

AUX_COLUMN = "_file"  # nom du fichier d'origine (utile pour un split par fichier)
LABEL_TEXT_COLUMN = "_label_text"


def available_memory_bytes() -> int | None:
    """Return available physical memory on Windows, when the API is available."""
    if not hasattr(ctypes, "windll"):
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_uint32), ("memory_load", ctypes.c_uint32),
            ("total", ctypes.c_uint64), ("available", ctypes.c_uint64),
            ("page_total", ctypes.c_uint64), ("page_available", ctypes.c_uint64),
            ("virtual_total", ctypes.c_uint64), ("virtual_available", ctypes.c_uint64),
            ("extended", ctypes.c_uint64), ("virtual_extended", ctypes.c_uint64),
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.available)
    return None


def memory_preflight(paths: list[Path], multiplier: float = 3.0) -> dict:
    """Estimate CSV loading pressure without loading any file."""
    total_bytes = sum(path.stat().st_size for path in paths if path.exists())
    available = available_memory_bytes()
    estimated = int(total_bytes * multiplier)
    return {
        "files": [str(path) for path in paths],
        "input_bytes": total_bytes,
        "estimated_peak_bytes": estimated,
        "available_memory_bytes": available,
        "sufficient_for_estimate": available is None or estimated < available,
        "multiplier": multiplier,
    }


def _read_one(path: Path, sample_frac: float | None, seed: int) -> pd.DataFrame:
    kwargs: dict = {"low_memory": False, "encoding_errors": "replace"}
    if sample_frac and sample_frac < 1.0:
        rng = np.random.default_rng(seed)
        kwargs["skiprows"] = lambda i: i > 0 and rng.random() > sample_frac
    df = pd.read_csv(path, **kwargs)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_csv_dir(directory: str | Path, sample_frac: float | None = None, seed: int = 0,
                 selected_files: list[str] | None = None) -> pd.DataFrame:
    """Concatène les CSV sélectionnés d'un dossier (ou tous si non spécifié)."""
    root = Path(directory)
    all_files = sorted(root.rglob("*.csv"))
    if selected_files is None:
        files = all_files
    else:
        wanted = {Path(name).name for name in selected_files}
        files = [f for f in all_files if f.name in wanted]
        missing = sorted(wanted - {f.name for f in files})
        if missing:
            raise FileNotFoundError(
                f"Fichier(s) source introuvable(s) dans {directory}: {missing}"
            )
    if not files:
        raise FileNotFoundError(f"Aucun fichier .csv trouvé dans {directory}")
    frames = []
    for f in files:
        df = _read_one(f, sample_frac, seed)
        df[AUX_COLUMN] = f.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def apply_label_mapping(df: pd.DataFrame, label_column: str,
                        mapping: dict[str, int]) -> pd.DataFrame:
    """Applique un mapping de labels explicite et refuse les labels inconnus."""
    out = df.copy()
    labels = out[label_column].astype(str).str.strip()
    unknown = sorted(set(labels) - set(mapping))
    if unknown:
        raise ValueError(
            f"Label(s) source inconnu(s): {unknown}. "
            f"Labels attendus: {sorted(mapping)}"
        )
    out[LABEL_TEXT_COLUMN] = labels
    out[label_column] = labels.map(mapping).astype("int64")
    return out


def clean_frame(df: pd.DataFrame, label_column: str, drop_columns: list[str],
                return_stats: bool = False):
    """Nettoie un DataFrame et peut retourner les diagnostics de chaque opération.

    Les NaN restants sont imputés plus tard (médiane calculée sur le train uniquement).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if label_column not in df.columns:
        raise KeyError(f"Colonne de label '{label_column}' introuvable. Colonnes: {list(df.columns)[:10]}...")

    df[label_column] = df[label_column].astype(str).str.strip()
    drop = [c for c in drop_columns if c in df.columns]
    df = df.drop(columns=drop)

    feats = [c for c in df.columns if c not in (label_column, AUX_COLUMN, LABEL_TEXT_COLUMN)]
    numeric = df[feats].apply(pd.to_numeric, errors="coerce")
    stats = {
        "rows_before": int(len(df)),
        "columns_before": int(len(df.columns)),
        "dropped_configured_columns": drop,
        "nan_by_column_before": {
            str(k): int(v) for k, v in numeric.isna().sum().items() if v
        },
        "positive_inf_by_column_before": {
            str(k): int(v) for k, v in numeric.eq(np.inf).sum().items() if v
        },
        "negative_inf_by_column_before": {
            str(k): int(v) for k, v in numeric.eq(-np.inf).sum().items() if v
        },
    }
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    invalid_columns = [c for c in feats if df[c].isna().all()]
    stats["entirely_invalid_columns"] = invalid_columns
    df = df.drop(columns=invalid_columns)

    feats = [c for c in df.columns if c not in (label_column, AUX_COLUMN, LABEL_TEXT_COLUMN)]
    constant_columns = [
        c for c in feats if df[c].nunique(dropna=True) <= 1
    ]
    stats["constant_columns"] = constant_columns
    before_duplicates = len(df)
    df = df.drop_duplicates(subset=feats + [label_column]).reset_index(drop=True)
    stats["duplicate_rows_removed"] = int(before_duplicates - len(df))
    stats["rows_after"] = int(len(df))
    stats["columns_after"] = int(len(df.columns))
    stats["nan_by_column_after"] = {
        str(k): int(v) for k, v in df[feats].isna().sum().items() if v
    }
    stats["note"] = "NaN et colonnes constantes sont conservés pour décision explicite en aval."
    gc.collect()
    return (df, stats) if return_stats else df
