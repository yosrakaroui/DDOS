"""Échantillonnage et séparation Train / Validation / Test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .loading import AUX_COLUMN


def drop_rare_classes(df: pd.DataFrame, label: str, min_rows: int) -> pd.DataFrame:
    counts = df[label].value_counts()
    keep = counts[counts >= min_rows].index
    return df[df[label].isin(keep)].reset_index(drop=True)


def cap_per_class(df: pd.DataFrame, label: str, max_rows: int, seed: int) -> pd.DataFrame:
    parts = [g.sample(n=min(len(g), max_rows), random_state=seed) for _, g in df.groupby(label)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _safe_split(df: pd.DataFrame, label: str, test_size: float, seed: int):
    try:
        return train_test_split(df, test_size=test_size, stratify=df[label], random_state=seed)
    except ValueError:  # classe trop rare pour une stratification
        return train_test_split(df, test_size=test_size, random_state=seed)


def stratified_split(df, label, val_size, test_size, seed):
    train_val, test = _safe_split(df, label, test_size, seed)
    train, val = _safe_split(train_val, label, val_size / (1.0 - test_size), seed)
    return train, val, test


def by_file_split(df, label, train_files, test_files, val_size, seed):
    """Test = fichiers (ou jours) réservés ; train/val tirés du reste."""
    if not test_files:
        raise ValueError("split.mode=by_file demande data.split.test_files non vide.")
    available = set(df[AUX_COLUMN].dropna().unique())
    requested = set(train_files) | set(test_files)
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"Fichier(s) inconnu(s) dans split.by_file: {unknown}")
    overlap = sorted(set(train_files) & set(test_files))
    if overlap:
        raise ValueError(f"Fichier(s) présents dans train et test: {overlap}")
    test = df[df[AUX_COLUMN].isin(test_files)]
    pool = df[df[AUX_COLUMN].isin(train_files)] if train_files else df[~df[AUX_COLUMN].isin(test_files)]
    if test.empty:
        raise ValueError("Le split by_file produit un test vide.")
    if pool.empty:
        raise ValueError("Le split by_file produit un pool train/validation vide.")
    expected = set(df[label].unique())
    missing_test = sorted(expected - set(test[label].unique()))
    if missing_test:
        raise ValueError(f"Classes absentes du test by_file: {missing_test}")
    train, val = _safe_split(pool, label, val_size, seed)
    if train.empty or val.empty:
        raise ValueError("Le split by_file produit un train ou une validation vide.")
    return train, val, test


def subsample_indices(y: np.ndarray, frac: float, seed: int, min_per_class: int = 5) -> np.ndarray:
    """Return deterministic stratified train indices for a low-data fraction."""
    if frac >= 1.0:
        return np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        n = min(len(ci), max(min_per_class, int(np.ceil(frac * len(ci)))))
        idx.append(rng.choice(ci, size=n, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx.astype(np.int64, copy=False)


def subsample_train(X: np.ndarray, y: np.ndarray, frac: float, seed: int, min_per_class: int = 5):
    """Sous-échantillon stratifié du train (étude low-data : 1 %, 5 %, 10 %...)."""
    idx = subsample_indices(y, frac, seed, min_per_class)
    return X[idx], y[idx]
