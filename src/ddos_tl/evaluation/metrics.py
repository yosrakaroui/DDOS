"""Métriques de classification multiclasses (sans dépendance à torch)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """Macro F1 sur les classes présentes dans y_true (même définition partout)."""
    labels = list(range(n_classes))
    _, _, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = s > 0
    return float(f[present].mean()) if present.any() else 0.0


def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = s > 0
    weighted_f1 = float(np.average(f, weights=s)) if s.sum() > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p[present].mean()) if present.any() else 0.0,
        "macro_recall": float(r[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f[present].mean()) if present.any() else 0.0,
        "weighted_f1": weighted_f1,
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
