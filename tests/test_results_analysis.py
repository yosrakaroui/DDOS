import numpy as np
import pandas as pd

from ddos_tl.evaluation.results_analysis import (
    audit_results_frame,
    negative_transfer_frame,
    per_class_frame,
    summarize_per_class,
    summarize_statistics,
)


def test_audit_accepts_complete_grid():
    rows = [
        {
            "strategy": strategy,
            "fraction": fraction,
            "seed": seed,
            "test_accuracy": 0.5,
            "test_macro_precision": 0.4,
            "test_macro_recall": 0.3,
            "test_macro_f1": 0.35,
            "test_weighted_f1": 0.5,
            "train_time_s": 1.0,
            "inference_us_per_sample": 2.0,
        }
        for strategy in ("scratch", "frozen", "partial", "finetune", "progressive")
        for fraction in (0.01, 0.05, 0.10, 1.0)
        for seed in (0, 1, 2)
    ]
    audit = audit_results_frame(pd.DataFrame(rows))
    assert audit["row_count_ok"]
    assert audit["missing_values"] == 0
    assert audit["duplicate_keys"] == 0


def test_summary_and_negative_transfer_are_deterministic():
    df = pd.DataFrame([
        {"strategy": "scratch", "fraction": 0.01, "seed": 0, "test_accuracy": 0.4,
         "test_macro_precision": 0.3, "test_macro_recall": 0.3, "test_macro_f1": 0.4,
         "test_weighted_f1": 0.5, "train_time_s": 1, "inference_us_per_sample": 2},
        {"strategy": "scratch", "fraction": 0.01, "seed": 1, "test_accuracy": 0.5,
         "test_macro_precision": 0.4, "test_macro_recall": 0.4, "test_macro_f1": 0.5,
         "test_weighted_f1": 0.6, "train_time_s": 3, "inference_us_per_sample": 4},
        {"strategy": "frozen", "fraction": 0.01, "seed": 0, "test_accuracy": 0.2,
         "test_macro_precision": 0.2, "test_macro_recall": 0.2, "test_macro_f1": 0.2,
         "test_weighted_f1": 0.3, "train_time_s": 1, "inference_us_per_sample": 2},
        {"strategy": "frozen", "fraction": 0.01, "seed": 1, "test_accuracy": 0.3,
         "test_macro_precision": 0.3, "test_macro_recall": 0.3, "test_macro_f1": 0.3,
         "test_weighted_f1": 0.4, "train_time_s": 3, "inference_us_per_sample": 4},
    ])
    summary = summarize_statistics(df)
    assert len(summary) == 2
    assert summary.loc[summary.strategy == "scratch", "test_macro_f1_mean"].iloc[0] == 0.45
    negative = negative_transfer_frame(df)
    assert negative.iloc[0]["delta_transfer_minus_scratch"] < 0


def test_per_class_flattening_preserves_support():
    records = [{
        "strategy": "scratch",
        "fraction": 0.01,
        "seed": 0,
        "test_metrics": {"per_class": {
            "UDPLag": {"precision": 0.1, "recall": 0.2, "f1": 0.13, "support": 8},
            "WebDDoS": {"precision": 0.3, "recall": 0.4, "f1": 0.34, "support": 8},
        }},
    }]
    flat = per_class_frame(records)
    summary = summarize_per_class(flat)
    assert set(flat["class"]) == {"UDPLag", "WebDDoS"}
    assert summary["support_mean"].tolist() == [8.0, 8.0]
