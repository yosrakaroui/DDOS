"""Prétraitement sans fuite de données : tout est ajusté sur le train uniquement.

Le préprocesseur est sérialisé en JSON (pas de pickle) pour être rechargé tel quel
dans l'application Web.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:
    """Imputation par la médiane + standardisation + bornage des valeurs extrêmes."""

    def __init__(self, feature_columns: list[str], clip: float = 10.0):
        self.feature_columns = list(feature_columns)
        self.clip = float(clip)
        self.median_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _to_array(self, df: pd.DataFrame) -> np.ndarray:
        sub = df.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors="coerce")
        X = sub.to_numpy(dtype=np.float64)
        return np.where(np.isfinite(X), X, np.nan)

    def missing_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.feature_columns if c not in df.columns]

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = self._to_array(df)
        med = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isnan(med), 0.0, med)
        Xf = np.where(np.isnan(X), self.median_, X)
        self.mean_ = Xf.mean(axis=0)
        std = Xf.std(axis=0)
        self.scale_ = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("Preprocessor non ajusté (appeler fit sur le train d'abord).")
        X = self._to_array(df)
        X = np.where(np.isnan(X), self.median_, X)
        Z = (X - self.mean_) / self.scale_
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "feature_columns": self.feature_columns,
            "clip": self.clip,
            "median": self.median_.tolist(),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        p = cls(d["feature_columns"], d.get("clip", 10.0))
        p.median_ = np.asarray(d["median"], dtype=np.float64)
        p.mean_ = np.asarray(d["mean"], dtype=np.float64)
        p.scale_ = np.asarray(d["scale"], dtype=np.float64)
        return p


class LabelMap:
    """Encodage des labels texte <-> entiers (ordre alphabétique, reproductible)."""

    def __init__(self, classes: list[str]):
        self.classes = list(classes)
        self._index = {c: i for i, c in enumerate(self.classes)}

    @classmethod
    def fit(cls, labels) -> "LabelMap":
        return cls(sorted(set(map(str, labels))))

    def encode(self, labels) -> np.ndarray:
        try:
            return np.asarray([self._index[str(x)] for x in labels], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"Label inconnu : {e}. Classes connues : {self.classes}") from e

    def decode(self, idx) -> list[str]:
        return [self.classes[int(i)] for i in idx]
