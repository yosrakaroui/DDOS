"""Controlled 2x2 runner with a safe dry-run path."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from ..config import resolve
from ..evaluation.metrics import compute_metrics, macro_f1
from ..inference import get_device, predict
from ..training.loops import class_weights, fit, make_loader, param_groups
from ..utils.io import load_json, save_json
from ..utils.seed import set_seed
from .two_by_two import (
    Cell,
    FRACTIONS,
    INITIALIZATIONS,
    N_CLASSES,
    SCALERS,
    SEEDS,
    build_initialization_bundle,
    build_grid,
    cell_inputs,
    instantiate_model,
    state_hashes,
    validate_grid,
)

OUTPUT_DIR = resolve("reports/results/phase_2x2_scaler_initialization")
MANIFEST_PATH = OUTPUT_DIR / "protocol_manifest.json"


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Load the fixed manifest and reject an incomplete or altered design."""
    if path != MANIFEST_PATH:
        raise ValueError(f"Manifest must be {MANIFEST_PATH}")
    manifest = load_json(path)
    validate_grid(manifest.get("grid", []))
    if manifest.get("grid_count") != 48 or manifest.get("class_count") != N_CLASSES:
        raise ValueError("Manifest dimensions do not match the controlled 2x2 design")
    return manifest


def _validate_shared_initialization(model, bundle, initialization: str, expected: dict) -> None:
    hashes = state_hashes(bundle)
    if hashes["shared_target_head"] != expected["shared_target_head"]:
        raise ValueError("Shared target-head hash does not match the manifest")
    key = "random_backbone" if initialization == "random" else "pretrained_backbone"
    model_backbone = {
        name: value for name, value in model.state_dict().items() if name.startswith("blocks.")
    }
    from .two_by_two import _state_hash
    if _state_hash(model_backbone) != expected[key]:
        raise ValueError("Model backbone hash does not match the manifest")
    model_head = {
        name: value for name, value in model.state_dict().items() if name.startswith("head.")
    }
    if _state_hash(model_head) != expected["shared_target_head"]:
        raise ValueError("Model head hash does not match the manifest")


def build_context(cfg: dict, manifest: dict) -> dict:
    """Load target artifacts and shared initial states without training."""
    from ..data.pipeline import load_splits

    data = load_splits(resolve(cfg["paths"]["processed"]), "target")
    if len(data["feature_columns"]) != 76 or len(data["class_names"]) != N_CLASSES:
        raise ValueError("Target artifacts do not match the 76-feature/18-class contract")
    pretrained_path = resolve(cfg["paths"]["models"]) / "pretrained_backbone.pt"
    bundles = {
        seed: build_initialization_bundle(
            seed, len(data["feature_columns"]), cfg["model"]["hidden"],
            cfg["model"]["dropout"], pretrained_path
        )
        for seed in SEEDS
    }
    return {"data": data, "bundles": bundles, "pretrained_path": pretrained_path}


def validate_run_configuration(row: dict, context: dict, cfg: dict, manifest: dict) -> dict:
    """Validate one cell, model state, scaler selection, and DataLoader."""
    seed = int(row["seed"])
    bundle = context["bundles"][seed]
    model = instantiate_model(
        bundle, row["initialization"], 76, cfg["model"]["hidden"], cfg["model"]["dropout"]
    )
    expected = manifest["initialization_hashes_by_seed"][str(seed)]
    _validate_shared_initialization(model, bundle, row["initialization"], expected)
    inputs = cell_inputs(context["data"], row["scaler"], float(row["fraction"]), seed)
    loader = make_loader(inputs["X_train"], inputs["y_train"], cfg["training"]["batch_size"], True, seed)
    return {"model": model, "inputs": inputs, "loader": loader, "seed": seed}


def validate_all(cfg: dict, manifest: dict) -> list[dict]:
    """Validate all 48 configurations without calling the training loop."""
    context = build_context(cfg, manifest)
    validations = []
    for row in manifest["grid"]:
        check = validate_run_configuration(row, context, cfg, manifest)
        validations.append({"run_id": row["run_id"], "n_train": len(check["inputs"]["y_train"])})
    return validations


def _assert_output_safety() -> None:
    if OUTPUT_DIR != resolve("reports/results/phase_2x2_scaler_initialization"):
        raise RuntimeError("2x2 output directory was changed")
    if not OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Missing fixed output directory: {OUTPUT_DIR}")


def run_one(row: dict, context: dict, cfg: dict, manifest: dict, device: torch.device) -> dict:
    """Execute one run; callers must explicitly opt into this function."""
    _assert_output_safety()
    result_path = OUTPUT_DIR / f"{row['run_id']}.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing 2x2 result: {result_path}")
    check = validate_run_configuration(row, context, cfg, manifest)
    model, inputs, loader = check["model"], check["inputs"], check["loader"]
    set_seed(check["seed"])
    train_cfg = cfg["training"]
    weights = class_weights(inputs["y_train"], N_CLASSES) if train_cfg["use_class_weights"] else None
    history, _ = fit(
        model, loader, inputs["X_val"], inputs["y_val"],
        epochs=train_cfg["epochs"],
        groups=param_groups(model, train_cfg["lr"], train_cfg["lr"]),
        weight_decay=train_cfg["weight_decay"],
        patience=train_cfg["patience"], weights=weights, device=device,
        n_classes=N_CLASSES, tag=row["cell"],
    )
    best_state = max(history, key=lambda item: item["val_macro_f1"])
    model.load_state_dict(deepcopy({
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }))
    val_f1 = macro_f1(inputs["y_val"], predict(model, inputs["X_val"], device), N_CLASSES)
    metrics = compute_metrics(
        inputs["y_test"],
        predict(model, inputs["X_test"], device),
        context["data"]["class_names"],
    )
    result = {
        **row,
        "n_train": len(inputs["y_train"]),
        "best_epoch": best_state["epoch"],
        "val_macro_f1": val_f1,
        "test_metrics": metrics,
        "scaler": row["scaler"],
        "output_directory": str(OUTPUT_DIR),
    }
    save_json(result, result_path)
    return result


def execute(cfg: dict, manifest: dict, device: torch.device | None = None) -> None:
    """Run all missing configurations; never touches legacy result paths."""
    _assert_output_safety()
    context = build_context(cfg, manifest)
    device = device or get_device(cfg["training"]["device"])
    for row in manifest["grid"]:
        run_one(row, context, cfg, manifest, device)


def dry_run(cfg: dict, manifest: dict) -> list[dict]:
    """Validate all configurations and print them without calling ``fit``."""
    _assert_output_safety()
    validations = validate_all(cfg, manifest)
    for row, check in zip(manifest["grid"], validations):
        print(f"{row['run_id']}: {row['initialization']} + {row['scaler']} "
              f"fraction={row['fraction']} seed={row['seed']} n_train={check['n_train']}")
    return validations
