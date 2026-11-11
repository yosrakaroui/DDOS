"""Stratégies comparées : scratch / frozen / partial / finetune / progressive."""
from __future__ import annotations

import time
from copy import deepcopy

import torch

from ..data.splitting import subsample_train
from ..evaluation.metrics import compute_metrics, macro_f1
from ..inference import get_device, predict
from ..models.mlp import TabularMLP
from ..utils.seed import set_seed
from .loops import class_weights, fit, make_loader, param_groups

STRATEGIES = ("scratch", "frozen", "partial", "finetune", "progressive")


def run_strategy(strategy: str, data: dict, cfg: dict, *, seed: int = 0, frac: float = 1.0,
                 pretrained_path=None, device=None, verbose: bool = False) -> dict:
    """Entraîne une stratégie sur (une fraction de) CIC-DDoS2019 et l'évalue sur le test."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Stratégie inconnue '{strategy}'. Choix : {STRATEGIES}")
    set_seed(seed)
    device = device or get_device(cfg["training"]["device"])
    t, x, m = cfg["training"], cfg["transfer"], cfg["model"]

    transfer = strategy != "scratch"
    if transfer and "feature_columns" in data and "X_transfer_train" not in data:
        raise ValueError(
            "Les artefacts cibles ne contiennent pas la variante normalisée avec "
            "le scaler source. Relancez prepare_data.py avant un transfert réel."
        )
    x_prefix = "X_transfer_" if transfer and "X_transfer_train" in data else "X_"
    X_tr, y_tr = subsample_train(data[f"{x_prefix}train"], data["y_train"], frac, seed)
    n_classes, in_dim = len(data["class_names"]), X_tr.shape[1]

    model = TabularMLP(in_dim, n_classes, m["hidden"], m["dropout"])
    if strategy != "scratch":
        if pretrained_path is None:
            raise ValueError("Cette stratégie demande des poids pré-entraînés (scripts/pretrain.py).")
        model.load_backbone(torch.load(pretrained_path, map_location="cpu", weights_only=True))
        # la tête est déjà une nouvelle couche adaptée à n_classes (cf. TabularMLP.__init__)

    weights = class_weights(y_tr, n_classes) if t["use_class_weights"] else None
    loader = make_loader(X_tr, y_tr, t["batch_size"], True, seed)
    common = dict(X_val=data[f"{x_prefix}val"], y_val=data["y_val"], weight_decay=t["weight_decay"],
                  patience=t["patience"], weights=weights, device=device, n_classes=n_classes,
                  verbose=verbose)
    lr, bb_lr = t["lr"], t["lr"] * x["finetune_lr_factor"]
    n_blocks = len(model.blocks)
    history: list[dict] = []
    best_val_f1 = -1.0
    best_state = None

    def stage(tag, epochs, head_lr, backbone_lr):
        nonlocal best_val_f1, best_state
        h, _ = fit(model, loader, epochs=epochs, groups=param_groups(model, head_lr, backbone_lr),
                   tag=tag, **common)
        history.extend(h)
        if h:
            stage_best = max(h, key=lambda row: row["val_macro_f1"])
            if stage_best["val_macro_f1"] > best_val_f1 + 1e-5:
                best_val_f1 = stage_best["val_macro_f1"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    t0 = time.perf_counter()
    if strategy == "scratch":                    # baseline : tout entraîné depuis zéro
        model.unfreeze_all()
        stage("scratch", t["epochs"], lr, lr)
    elif strategy == "frozen":                   # backbone entièrement gelé, seule la tête apprend
        model.freeze_blocks(n_blocks)
        stage("head", t["epochs"], lr, lr)
    elif strategy == "partial":                  # les k premiers blocs gelés, le reste est affiné
        model.freeze_blocks(min(x["partial_freeze_blocks"], n_blocks))
        stage("partial", t["epochs"], lr, bb_lr)
    elif strategy == "finetune":                 # 1) tête seule  2) tout débloqué, lr faible
        model.freeze_blocks(n_blocks)
        stage("head", x["head_epochs"], lr, lr)
        model.unfreeze_all()
        stage("finetune", t["epochs"], bb_lr, bb_lr)
    elif strategy == "progressive":              # dégel couche par couche, du haut vers le bas
        model.freeze_blocks(n_blocks)
        stage("head", x["head_epochs"], lr, lr)
        for k in range(n_blocks - 1, -1, -1):
            model.freeze_blocks(k)
            stage(f"unfreeze_from_block_{k}", x["stage_epochs"], bb_lr, bb_lr)
    train_time = time.perf_counter() - t0

    if best_state is None:
        raise RuntimeError("Aucun état de modèle n'a été produit pendant l'entraînement.")
    model.load_state_dict(deepcopy(best_state))
    val_f1 = macro_f1(data["y_val"], predict(model, data[f"{x_prefix}val"], device), n_classes)
    t1 = time.perf_counter()
    y_pred = predict(model, data[f"{x_prefix}test"], device)
    inference_us = (time.perf_counter() - t1) / max(len(y_pred), 1) * 1e6

    return {
        "model": model, "history": history, "strategy": strategy, "fraction": frac, "seed": seed,
        "n_train": int(len(y_tr)), "val_macro_f1": val_f1,
        "best_epoch": max(history, key=lambda row: row["val_macro_f1"])["epoch"],
        "test_metrics": compute_metrics(data["y_test"], y_pred, data["class_names"]),
        "scaler_type": data.get("preprocessing_metadata", {}).get(
            "transfer" if transfer else "scratch", {}
        ).get("scaler_type", "standard"),
        "scaler_fit_dataset": data.get("preprocessing_metadata", {}).get(
            "transfer" if transfer else "scratch", {}
        ).get("scaler_fit_dataset", "source" if transfer else "target"),
        "scaler_fit_split": "train",
        "clipping": data.get("preprocessing_metadata", {}).get(
            "transfer" if transfer else "scratch", {}
        ).get("clipping"),
        "scaler_mean": data.get("preprocessing_metadata", {}).get(
            "transfer" if transfer else "scratch", {}
        ).get("scaler_mean"),
        "scaler_scale": data.get("preprocessing_metadata", {}).get(
            "transfer" if transfer else "scratch", {}
        ).get("scaler_scale"),
        "feature_columns": data.get("feature_columns", []),
        "preprocessing_strategy": data.get("preprocessing_metadata", {}).get(
            "transfer" if transfer else "scratch", {}
        ).get("preprocessing_strategy"),
        "train_time_s": train_time, "inference_us_per_sample": inference_us,
    }
