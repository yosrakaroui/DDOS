"""Phase 4 : pré-entraînement du backbone sur le dataset SOURCE (ex. CIC-IDS2017).

Usage : python scripts/pretrain.py [--config configs/default.yaml]
Sorties : <models>/pretrained_backbone.pt (poids du backbone seulement)
          <models>/pretrained_meta.json  (origine des poids : preuve du pré-entraînement)
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config, resolve
from ddos_tl.data.pipeline import load_splits
from ddos_tl.evaluation.metrics import compute_metrics
from ddos_tl.inference import get_device, predict
from ddos_tl.models.mlp import TabularMLP
from ddos_tl.training.loops import class_weights, fit, make_loader, param_groups
from ddos_tl.utils.io import save_json
from ddos_tl.utils.seed import set_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    t, m = cfg["training"], cfg["model"]
    device = get_device(t["device"])

    processed = resolve(cfg["paths"]["processed"])
    src = load_splits(processed, "source")
    tgt = load_splits(processed, "target")
    if src["feature_columns"] != tgt["feature_columns"]:
        raise SystemExit("Features source != cible : relance scripts/prepare_data.py.")

    n_classes = len(src["class_names"])
    model = TabularMLP(src["X_train"].shape[1], n_classes, m["hidden"], m["dropout"])
    loader = make_loader(src["X_train"], src["y_train"], t["batch_size"], True, cfg["seed"])
    weights = class_weights(src["y_train"], n_classes) if t["use_class_weights"] else None
    started = time.perf_counter()
    history, best = fit(model, loader, src["X_val"], src["y_val"], epochs=t["epochs"],
                        groups=param_groups(model, t["lr"], t["lr"]), weight_decay=t["weight_decay"],
                        patience=t["patience"], weights=weights, device=device, n_classes=n_classes,
                        tag="pretrain", verbose=True)
    duration_s = time.perf_counter() - started
    best_epoch = max(history, key=lambda row: row["val_macro_f1"])["epoch"]

    metrics = compute_metrics(src["y_test"], predict(model, src["X_test"], device), src["class_names"])
    out = resolve(cfg["paths"]["models"])
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "backbone_state_dict": model.backbone_state_dict(),
        "arch": model.arch,
        "feature_columns": src["feature_columns"],
        "class_names": src["class_names"],
        "best_epoch": best_epoch,
        "validation_macro_f1": best,
    }
    torch.save(checkpoint, out / "pretrained_checkpoint.pt")
    torch.save(model.backbone_state_dict(), out / "pretrained_backbone.pt")
    metadata = {
        "source_dataset": "CIC-IDS2017",
        "source_path": str(cfg["paths"]["source_raw"]),
        "source_files": cfg["data"].get("source_files"),
        "provenance": cfg["data"].get("provenance", {}).get("source"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg["seed"], "hidden": m["hidden"], "dropout": m["dropout"],
        "n_features": len(src["feature_columns"]), "feature_columns": src["feature_columns"],
        "source_classes": src["class_names"],
        "label_mapping": src.get("label_mapping", cfg["data"].get("source_label_mapping")),
        "feature_alignment": src.get("feature_alignment"),
        "split_metadata": src.get("split_metadata"),
        "validation_metrics": {"macro_f1": best, "best_epoch": best_epoch},
        "test_metrics": metrics,
        "training": t,
        "duration_s": duration_s,
        "history": history,
        "preprocessing_metadata": src.get("preprocessing_metadata", {}),
    }
    save_json(metadata, out / "pretrained_meta.json")
    save_json({
        "experiment": "source_pretraining",
        "dataset": "CIC-IDS2017",
        "best_epoch": best_epoch,
        "duration_s": duration_s,
        "validation_metrics": metadata["validation_metrics"],
        "test_metrics": metrics,
        "history": history,
    }, out / "pretrain_report.json")
    print(f"Backbone pré-entraîné sauvegardé. Macro F1 test (source) = {metrics['macro_f1']:.4f}")
