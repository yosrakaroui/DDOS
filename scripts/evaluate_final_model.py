"""Evaluate the exported production model without training or model selection."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_tl.config import load_config, resolve  # noqa: E402
from ddos_tl.data.pipeline import load_splits  # noqa: E402
from ddos_tl.evaluation.metrics import compute_metrics  # noqa: E402
from ddos_tl.evaluation.plots import plot_confusion_matrix  # noqa: E402
from ddos_tl.inference import predict_proba  # noqa: E402
from ddos_tl.models.mlp import TabularMLP  # noqa: E402
from ddos_tl.utils.io import load_json, save_json  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = load_config("configs/default.yaml")
    models = resolve(cfg["paths"]["models"])
    output = resolve("reports/results/final_model")
    model_path, meta_path = models / "best_model.pt", models / "best_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Exported production model artifacts are missing")
    meta = load_json(meta_path)
    if len(meta["feature_columns"]) != 76 or len(meta["class_names"]) != 18:
        raise ValueError("Production metadata does not match the 76-feature/18-class contract")
    if meta["checkpoint_sha256"] != sha256(model_path):
        raise ValueError("Production checkpoint hash does not match best_meta.json")
    model = TabularMLP(
        meta["arch"]["in_dim"], len(meta["class_names"]),
        meta["arch"]["hidden"], meta["arch"]["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    if model.head.out_features != 18:
        raise ValueError("Production model does not have 18 outputs")
    data = load_splits(resolve(cfg["paths"]["processed"]), "target")
    if data["feature_columns"] != meta["feature_columns"]:
        raise ValueError("Production metadata and target feature order differ")
    inference_start = time.perf_counter()
    probabilities = predict_proba(model, data["X_test"], torch.device("cpu"))
    inference_seconds = time.perf_counter() - inference_start
    if probabilities.shape != (len(data["X_test"]), 18):
        raise ValueError("Invalid production probability shape")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Invalid production probabilities")
    predictions = probabilities.argmax(axis=1)
    metrics = compute_metrics(data["y_test"], predictions, data["class_names"])
    output.mkdir(parents=True, exist_ok=True)
    save_json({
        "model": "models/best_model.pt",
        "metadata": "models/best_meta.json",
        "checkpoint_sha256": sha256(model_path),
        "evaluation_split": "target/test",
        "selection_split": "target/val",
        "selection_metric": "validation_macro_f1",
        "test_metrics": metrics,
        "probability_checks": {
            "shape": list(probabilities.shape),
            "finite": bool(np.isfinite(probabilities).all()),
            "rows_sum_to_one": bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)),
        },
        "inference_time_s": inference_seconds,
        "inference_us_per_sample": inference_seconds / max(len(data["X_test"]), 1) * 1e6,
    }, output / "evaluation.json")
    np.save(output / "confusion_matrix.npy", np.asarray(metrics["confusion_matrix"], dtype=np.int64))
    with (output / "per_class_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics["per_class"], handle, indent=2, ensure_ascii=False)
    plot_confusion_matrix(
        metrics["confusion_matrix"], data["class_names"],
        output / "confusion_matrix.png",
    )
    print(json.dumps({
        "test_accuracy": metrics["accuracy"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall": metrics["macro_recall"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"],
        "udplag": metrics["per_class"]["UDPLag"],
        "webddos": metrics["per_class"]["WebDDoS"],
    }, indent=2))


if __name__ == "__main__":
    main()
