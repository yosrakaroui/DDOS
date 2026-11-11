"""Boucle d'entraînement générique (early stopping sur le Macro F1 de validation)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.metrics import macro_f1
from ..inference import predict


def make_loader(X, y, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long))
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen if shuffle else None)


def class_weights(y: np.ndarray, n_classes: int, max_weight: float = 10.0) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1.0)), 0.0)
    w = w / w[w > 0].mean()
    return torch.tensor(np.clip(w, 0.0, max_weight), dtype=torch.float32)


def param_groups(model, head_lr: float, backbone_lr: float) -> list[dict]:
    """Learning rates différenciés ; les paramètres gelés sont exclus de l'optimiseur."""
    head = [p for p in model.head.parameters() if p.requires_grad]
    back = [p for b in model.blocks for p in b.parameters() if p.requires_grad]
    groups = []
    if head:
        groups.append({"params": head, "lr": head_lr})
    if back:
        groups.append({"params": back, "lr": backbone_lr})
    return groups


def fit(model, train_loader, X_val, y_val, *, epochs: int, groups: list[dict], weight_decay: float,
        patience: int, weights, device, n_classes: int, tag: str = "", verbose: bool = False):
    """Entraîne `model` et recharge les meilleurs poids (selon le Macro F1 de validation)."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    optimizer = torch.optim.Adam(groups, weight_decay=weight_decay)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)

        val_f1 = macro_f1(y_val, predict(model, X_val, device), n_classes)
        history.append({"stage": tag, "epoch": epoch, "train_loss": total / max(n, 1), "val_macro_f1": val_f1})
        if verbose:
            print(f"[{tag}] epoch {epoch:02d} loss={total / max(n, 1):.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1 + 1e-5:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return history, best_f1
