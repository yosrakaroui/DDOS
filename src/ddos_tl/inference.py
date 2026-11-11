"""Prédiction avec un TabularMLP (utilisé par l'entraînement et par l'application Web)."""
from __future__ import annotations

import numpy as np
import torch


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


@torch.no_grad()
def predict_proba(model, X: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    model.to(device)
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, model.head.out_features), dtype=np.float32)


def predict(model, X: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    return predict_proba(model, X, device, batch_size).argmax(axis=1)
