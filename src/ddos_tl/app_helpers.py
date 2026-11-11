"""Pure helpers for the Streamlit dashboard."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from .data.features import apply_alignment
from .data.preprocessing import Preprocessor
from .evaluation.metrics import compute_metrics


def load_uploaded_csv(uploaded, max_rows: int, alignment: dict | None = None) -> pd.DataFrame:
    """Read, normalize headers and validate an uploaded CICFlowMeter CSV."""
    if uploaded is None:
        raise ValueError("Aucun fichier CSV n'a été fourni.")
    content = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
    if not content:
        raise ValueError("Le fichier CSV est vide.")
    try:
        frame = pd.read_csv(BytesIO(content), low_memory=False)
    except Exception as exc:
        raise ValueError(f"Lecture du CSV impossible : {exc}") from exc
    if frame.empty:
        raise ValueError("Le fichier CSV ne contient aucune ligne.")
    if len(frame) > max_rows:
        raise ValueError(
            f"Le fichier contient {len(frame):,} lignes, au-delà de la limite "
            f"configurée de {max_rows:,}. Aucune ligne n'a été supprimée."
        )
    frame.columns = [str(column).strip() for column in frame.columns]
    if alignment:
        frame = apply_alignment(frame, alignment, "target")
    return frame


def validate_input_columns(frame: pd.DataFrame, prep: Preprocessor) -> list[str]:
    """Return missing model columns without inventing or dropping columns."""
    return prep.missing_columns(frame)


def transform_for_inference(frame: pd.DataFrame, prep: Preprocessor) -> tuple[np.ndarray, dict]:
    """Apply the saved preprocessor and report invalid values before imputation."""
    numeric = frame.reindex(columns=prep.feature_columns).apply(pd.to_numeric, errors="coerce")
    invalid = (~np.isfinite(numeric.to_numpy(dtype=np.float64))).sum(axis=0)
    invalid_by_column = {
        column: int(count)
        for column, count in zip(prep.feature_columns, invalid)
        if count
    }
    transformed = prep.transform(frame)
    if not np.isfinite(transformed).all():
        raise ValueError("Le preprocessing a produit des valeurs non finies.")
    return transformed, {
        "invalid_cells": int(invalid.sum()),
        "invalid_by_column": invalid_by_column,
    }


def confidence_level(value: float) -> str:
    if value >= 0.80:
        return "Élevée"
    if value >= 0.50:
        return "Moyenne"
    return "Faible"


def build_prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray,
                           class_names: list[str]) -> pd.DataFrame:
    """Build a stable prediction table with all class probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != (len(frame), len(class_names)):
        raise ValueError("Dimension de probabilités incompatible avec les données.")
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("Probabilités non finies ou non normalisées.")
    indices = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    result = pd.DataFrame({
        "Flow index": frame.index.to_numpy(),
        "Classe prédite": np.asarray(class_names)[indices],
        "Confidence": confidence,
        "Niveau": [confidence_level(value) for value in confidence],
    }, index=frame.index)
    for index, name in enumerate(class_names):
        result[f"P({name})"] = probabilities[:, index]
    return result


def evaluate_labeled_frame(frame: pd.DataFrame, prediction_indices: np.ndarray,
                           class_names: list[str], label_column: str) -> tuple[dict | None, int]:
    """Evaluate only recognized uploaded labels; predictions remain model outputs."""
    if label_column not in frame.columns:
        return None, 0
    labels = frame[label_column].astype(str).str.strip()
    mapping = {name: index for index, name in enumerate(class_names)}
    known = labels.isin(mapping)
    if not known.any():
        return None, int((~known).sum())
    y_true = labels[known].map(mapping).to_numpy(dtype=np.int64)
    y_pred = np.asarray(prediction_indices)[known.to_numpy()]
    return compute_metrics(y_true, y_pred, class_names), int((~known).sum())
