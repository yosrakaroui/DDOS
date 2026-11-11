"""Train and export the validation-selected multiclass model for Streamlit.

The script is inert unless ``--execute`` is supplied. Test metrics are recorded
after model selection but are never used to choose the configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_tl.config import load_config, resolve  # noqa: E402
from ddos_tl.data.pipeline import load_splits  # noqa: E402
from ddos_tl.inference import get_device  # noqa: E402
from ddos_tl.training.strategies import run_strategy  # noqa: E402
from ddos_tl.utils.io import load_json, save_json  # noqa: E402
from ddos_tl.utils.seed import set_seed  # noqa: E402

STRATEGY = "scratch"
FRACTION = 1.0
SEED = 0
EXPECTED_FEATURES = 76
EXPECTED_CLASSES = 18
EXPECTED_HIDDEN = [256, 128, 64]
EXPECTED_DROPOUT = 0.2
EXPECTED_TRAINING = {
    "epochs": 30,
    "batch_size": 512,
    "lr": 0.001,
    "weight_decay": 0.00001,
    "patience": 5,
    "use_class_weights": True,
}


def validation_selection() -> dict:
    """Select the final protocol from validation Macro F1 only."""
    records = []
    for path in sorted((ROOT / "reports" / "results" / "runs").glob("*.json")):
        record = load_json(path)
        records.append(record)
    if len(records) != 60:
        raise ValueError(f"Expected 60 historical records, found {len(records)}")
    groups = {}
    for record in records:
        key = (record["strategy"], float(record["fraction"]))
        groups.setdefault(key, []).append(float(record["val_macro_f1"]))
    means = {
        key: float(np.mean(values))
        for key, values in groups.items()
        if len(values) == 3 and np.isfinite(values).all()
    }
    if len(means) != 20:
        raise ValueError("Historical validation grid is incomplete")
    best_strategy, best_fraction = max(means, key=means.get)
    return {
        "strategy": best_strategy,
        "fraction": best_fraction,
        "validation_macro_f1_mean": means[(best_strategy, best_fraction)],
        "selection_metric": "validation_macro_f1_mean",
        "test_metrics_used_for_selection": False,
        "group_means": {
            f"{strategy}_f{fraction}": value
            for (strategy, fraction), value in sorted(means.items())
        },
    }


def validate_data_contract(data: dict) -> None:
    if len(data["feature_columns"]) != EXPECTED_FEATURES:
        raise ValueError("Target data does not contain exactly 76 features")
    if len(data["class_names"]) != EXPECTED_CLASSES:
        raise ValueError("Target data does not contain exactly 18 classes")
    if data["preprocessing_metadata"].get("scaler_fit_dataset") != "target":
        raise ValueError("Final model must use the target scaler")
    if data["preprocessing_metadata"].get("scaler_fit_split") != "train":
        raise ValueError("Final scaler must be fitted on target/train")
    if data["preprocessor"]["feature_columns"] != data["feature_columns"]:
        raise ValueError("Preprocessor feature order differs from target feature order")
    if data["preprocessor"].get("clip") != 10.0:
        raise ValueError("Unexpected clipping configuration")
    if not np.isfinite(np.asarray(data["X_train"])).all():
        raise ValueError("Target training features contain non-finite values")


def validate_final_protocol(data: dict, cfg: dict, selection: dict) -> dict:
    """Prove that the loaded artifacts match the historical Scratch 100% run."""
    if selection["strategy"] != STRATEGY or selection["fraction"] != FRACTION:
        raise ValueError(f"Unexpected validation-selected protocol: {selection}")
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    if list(model_cfg["hidden"]) != EXPECTED_HIDDEN or float(model_cfg["dropout"]) != EXPECTED_DROPOUT:
        raise ValueError("Model architecture differs from the historical protocol")
    for key, expected in EXPECTED_TRAINING.items():
        actual = training_cfg[key]
        if actual != expected:
            raise ValueError(f"Training parameter {key} differs from the historical protocol")
    source = load_json(ROOT / "reports" / "results" / "runs" / "scratch_f1.0_s0.json")
    if source["n_train"] != len(data["y_train"]):
        raise ValueError("Loaded target/train size differs from Scratch 100% seed 0")
    if source["feature_columns"] != data["feature_columns"]:
        raise ValueError("Loaded feature order differs from Scratch 100% seed 0")
    if source["scaler_fit_dataset"] != "target" or source["scaler_fit_split"] != "train":
        raise ValueError("Historical Scratch run does not use target/train preprocessing")
    if source["clipping"] != data["preprocessing_metadata"]["clipping"]:
        raise ValueError("Clipping differs from the historical Scratch run")
    return source


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_model(cfg: dict, selection: dict) -> None:
    """Run the explicitly authorized final training and export its best state."""
    if selection["strategy"] != STRATEGY or selection["fraction"] != FRACTION:
        raise ValueError(f"Unexpected validation-selected protocol: {selection}")
    models_dir = resolve(cfg["paths"]["models"])
    model_path = models_dir / "best_model.pt"
    meta_path = models_dir / "best_meta.json"
    if model_path.exists() or meta_path.exists():
        raise FileExistsError("Refusing to overwrite an existing production model artifact")

    data = load_splits(resolve(cfg["paths"]["processed"]), "target")
    validate_data_contract(data)
    historical_run = validate_final_protocol(data, cfg, selection)
    set_seed(SEED)
    result = run_strategy(
        STRATEGY,
        data,
        cfg,
        seed=SEED,
        frac=FRACTION,
        pretrained_path=None,
        device=get_device(cfg["training"]["device"]),
        verbose=True,
    )
    if result["model"].head.out_features != EXPECTED_CLASSES:
        raise ValueError("Exported model does not have 18 output classes")
    if result["val_macro_f1"] <= 0:
        raise RuntimeError("Validation Macro F1 is invalid")

    models_dir.mkdir(parents=True, exist_ok=True)
    temp_model_path = models_dir / ".best_model.pt.tmp"
    temp_meta_path = models_dir / ".best_meta.json.tmp"
    torch.save(result["model"].state_dict(), temp_model_path)
    metrics = result["test_metrics"]
    preprocessing = data["preprocessing_metadata"]
    metadata = {
        "strategy": STRATEGY,
        "selection": selection,
        "selected_seed": SEED,
        "arch": result["model"].arch,
        "training_config": cfg["training"],
        "feature_columns": data["feature_columns"],
        "class_names": data["class_names"],
        "preprocessor": data["preprocessor"],
        "feature_alignment": data.get("feature_alignment"),
        "scaler_type": preprocessing["scaler_type"],
        "scaler_fit_dataset": preprocessing["scaler_fit_dataset"],
        "scaler_fit_split": preprocessing["scaler_fit_split"],
        "clipping": preprocessing["clipping"],
        "preprocessing_strategy": preprocessing["preprocessing_strategy"],
        "validation_macro_f1": result["val_macro_f1"],
        "best_epoch": result["best_epoch"],
        "test_metrics": {
            key: metrics[key]
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
        },
        "test_metrics_role": "post_selection_evaluation_only",
        "historical_reference": "reports/results/runs/scratch_f1.0_s0.json",
        "historical_reference_val_macro_f1": historical_run["val_macro_f1"],
    }
    metadata["checkpoint_sha256"] = _sha256(temp_model_path)
    save_json(metadata, temp_meta_path)
    os.replace(temp_model_path, model_path)
    os.replace(temp_meta_path, meta_path)
    loaded_state = torch.load(model_path, map_location="cpu", weights_only=True)
    if _sha256(model_path) != metadata["checkpoint_sha256"] or not loaded_state:
        raise RuntimeError("Exported checkpoint failed post-write verification")
    print(f"Exported {model_path}")
    print(f"Exported {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--execute", action="store_true",
                        help="explicitly authorize the final training and export")
    args = parser.parse_args()
    cfg = load_config(args.config)
    selection = validation_selection()
    print(json.dumps(selection, indent=2))
    if not args.execute:
        print("No training launched. Re-run with --execute after explicit authorization.")
        return
    export_model(cfg, selection)


if __name__ == "__main__":
    main()
