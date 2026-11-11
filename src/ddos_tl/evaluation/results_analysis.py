"""Pure helpers for analysing completed experiment results."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


GRID_FRACTIONS = (0.01, 0.05, 0.10, 1.0)
GRID_SEEDS = (0, 1, 2)
GRID_STRATEGIES = ("scratch", "frozen", "partial", "finetune", "progressive")
METRICS = (
    "test_accuracy",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "test_weighted_f1",
)


def audit_results_frame(df: pd.DataFrame) -> dict:
    """Validate the completed grid without changing the input frame."""
    required = {"strategy", "fraction", "seed", *METRICS, "train_time_s", "inference_us_per_sample"}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing result columns: {missing_columns}")
    keys = df[["strategy", "fraction", "seed"]]
    numeric = df.select_dtypes(include="number")
    expected = len(GRID_STRATEGIES) * len(GRID_FRACTIONS) * len(GRID_SEEDS)
    return {
        "rows": int(len(df)),
        "expected_rows": expected,
        "row_count_ok": len(df) == expected,
        "missing_values": int(df.isna().sum().sum()),
        "nonfinite_values": int((~np.isfinite(numeric.to_numpy())).sum()),
        "duplicate_keys": int(keys.duplicated().sum()),
        "unexpected_strategies": sorted(set(df["strategy"]) - set(GRID_STRATEGIES)),
        "unexpected_fractions": sorted(set(df["fraction"]) - set(GRID_FRACTIONS)),
        "unexpected_seeds": sorted(set(df["seed"]) - set(GRID_SEEDS)),
    }


def summarize_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean and sample standard deviation by strategy and fraction."""
    grouped = df.groupby(["strategy", "fraction"], sort=True)
    out = grouped[list(METRICS) + ["train_time_s", "inference_us_per_sample"]].agg(["mean", "std"])
    out.columns = [f"{column}_{stat}" for column, stat in out.columns]
    return out.reset_index()


def load_run_records(runs_dir: str | Path) -> list[dict]:
    """Load detailed run JSON records in deterministic filename order."""
    records = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        record["_path"] = str(path)
        records.append(record)
    return records


def per_class_frame(records: list[dict]) -> pd.DataFrame:
    """Flatten per-class test metrics, preserving every class and support."""
    rows = []
    for record in records:
        for class_name, metrics in record["test_metrics"]["per_class"].items():
            rows.append({
                "strategy": record["strategy"],
                "fraction": record["fraction"],
                "seed": record["seed"],
                "class": class_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "support": metrics["support"],
            })
    return pd.DataFrame(rows)


def summarize_per_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate class metrics and retain mean support across seeds."""
    grouped = frame.groupby(["strategy", "fraction", "class"], sort=True)
    summary = grouped[["precision", "recall", "f1", "support"]].agg(["mean", "std"])
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return summary.reset_index()


def negative_transfer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each transfer strategy with Scratch at each fraction."""
    scratch = df[df["strategy"] == "scratch"].groupby("fraction")["test_macro_f1"].agg(
        scratch_mean="mean", scratch_std="std"
    )
    transfer = df[df["strategy"] != "scratch"].groupby(["strategy", "fraction"])["test_macro_f1"].agg(
        transfer_mean="mean", transfer_std="std"
    ).reset_index()
    out = transfer.merge(scratch.reset_index(), on="fraction", how="left")
    out["delta_transfer_minus_scratch"] = out["transfer_mean"] - out["scratch_mean"]
    out["interpretation"] = np.where(
        out["delta_transfer_minus_scratch"] > 0,
        "observed positive transfer",
        np.where(out["delta_transfer_minus_scratch"] < 0, "observed negative transfer", "no mean difference"),
    )
    return out.sort_values(["fraction", "strategy"]).reset_index(drop=True)
