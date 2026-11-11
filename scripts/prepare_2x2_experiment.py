"""Validate and materialize the 48-run 2x2 manifest; never trains."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config, resolve
from ddos_tl.data.pipeline import load_splits
from ddos_tl.experiments.two_by_two import (
    FRACTIONS,
    SEEDS,
    build_grid,
    build_initialization_bundle,
    cell_inputs,
    state_hashes,
    validate_grid,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 2x2 manifest without training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="reports/results/phase_2x2_scaler_initialization")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = resolve(cfg["paths"]["processed"])
    models = resolve(cfg["paths"]["models"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    data = load_splits(processed, "target")
    if len(data["feature_columns"]) != 76:
        raise ValueError("Expected exactly 76 target features")
    if len(data["class_names"]) != 18:
        raise ValueError("Expected exactly 18 target classes")
    if data["feature_columns"] != data["preprocessing_metadata"]["feature_columns"]:
        raise ValueError("Target feature metadata order is inconsistent")
    if "X_transfer_train" not in data:
        raise ValueError("Source-scaled target arrays are missing")

    grid = build_grid(FRACTIONS, SEEDS)
    validate_grid(grid)
    bundles = {
        seed: build_initialization_bundle(
            seed, len(data["feature_columns"]), cfg["model"]["hidden"],
            cfg["model"]["dropout"], models / "pretrained_backbone.pt"
        )
        for seed in SEEDS
    }
    for seed, bundle in bundles.items():
        for fraction in FRACTIONS:
            target_inputs = cell_inputs(data, "target", fraction, seed)
            source_inputs = cell_inputs(data, "source", fraction, seed)
            if not (target_inputs["subset_indices"] == source_inputs["subset_indices"]).all():
                raise ValueError(f"Subset indices differ for fraction {fraction}, seed {seed}")
            if not (target_inputs["y_train"] == source_inputs["y_train"]).all():
                raise ValueError(f"Subset labels differ for fraction {fraction}, seed {seed}")

    protected = [
        models / "pretrained_backbone.pt",
        models / "pretrained_checkpoint.pt",
        processed / "source_meta.json",
        processed / "target_meta.json",
        processed / "source_splits.npz",
        processed / "target_splits.npz",
        resolve(cfg["paths"]["results"]) / "experiments.csv",
    ]
    manifest = {
        "training_launched": False,
        "unsupervised_target_adaptation": (
            "target scaler is fit on all target/train features before supervised subset selection"
        ),
        "feature_count": len(data["feature_columns"]),
        "class_count": len(data["class_names"]),
        "feature_columns": data["feature_columns"],
        "grid_count": len(grid),
        "grid": grid,
        "shared_training_protocol": {
            "optimizer": "Adam",
            "learning_rate": cfg["training"]["lr"],
            "weight_decay": cfg["training"]["weight_decay"],
            "batch_size": cfg["training"]["batch_size"],
            "epochs": cfg["training"]["epochs"],
            "patience": cfg["training"]["patience"],
            "use_class_weights": cfg["training"]["use_class_weights"],
            "selection_metric": "validation_macro_f1",
            "test_evaluation": "after_validation_selection_only",
            "dataloader_generator_seed": "same experiment seed for every cell",
        },
        "scalers": {
            "target": data["preprocessing_metadata"]["scratch"],
            "source": data["preprocessing_metadata"]["transfer"],
            "clip": data["preprocessing_metadata"]["clipping"],
            "per_feature_clipping_rates": (
                "Not recoverable from serialized post-clipping arrays; requires pre-clipping values."
            ),
        },
        "initialization_hashes_by_seed": {
            str(seed): state_hashes(bundle) for seed, bundle in bundles.items()
        },
        "protected_artifacts_current_hashes": {
            str(path): sha256(path) for path in protected if path.exists()
        },
    }
    (output / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Validated {len(grid)} configurations; no training launched.")


if __name__ == "__main__":
    main()
