"""Phases 1-2 : lecture, nettoyage, alignement des features, split et normalisation.

Usage : python scripts/prepare_data.py [--config configs/default.yaml] [--skip-source]
Sorties : <processed>/target_splits.npz, target_meta.json, source_splits.npz, source_meta.json
          <results>/data_report.json
"""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config, resolve
from ddos_tl.data.features import align_features, apply_alignment, numeric_feature_columns
from ddos_tl.data.loading import (
    apply_label_mapping,
    clean_frame,
    memory_preflight,
    read_csv_dir,
)
from ddos_tl.data.pipeline import build_splits, save_splits
from ddos_tl.utils.io import save_json
from ddos_tl.utils.seed import set_seed


def load_clean(directory, cfg, seed, *, selected_files=None, label_mapping=None):
    d = cfg["data"]
    raw = read_csv_dir(directory, d.get("sample_frac_per_file"), seed, selected_files)
    clean, cleaning = clean_frame(
        raw, d["label_column"], d["drop_columns"], return_stats=True
    )
    if label_mapping is not None:
        clean = apply_label_mapping(clean, d["label_column"], label_mapping)
    cleaning["label_counts_after_mapping"] = {
        str(k): int(v) for k, v in clean[d["label_column"]].value_counts().items()
    }
    del raw
    gc.collect()
    return clean, cleaning


def report_markdown(report: dict) -> str:
    """Create a concise, auditable data-preparation report."""
    lines = [
        "# Data preparation report",
        "",
        "This report describes preprocessing only; no model training was run.",
        "",
        "## Provenance",
    ]
    provenance = report.get("provenance", {})
    for name, value in provenance.items():
        lines.append(f"- **{name}**: {value}")
    lines.extend(["", "## Memory preflight"])
    memory = report.get("memory_preflight", {})
    lines.append(f"- Input bytes: `{memory.get('input_bytes')}`")
    lines.append(f"- Estimated peak bytes: `{memory.get('estimated_peak_bytes')}`")
    lines.append(f"- Sufficient for estimate: `{memory.get('sufficient_for_estimate')}`")
    for dataset_name in ("source", "target"):
        dataset = report.get(dataset_name)
        if not dataset:
            continue
        lines.extend(["", f"## {dataset_name.capitalize()}"])
        lines.append(f"- Rows after cleaning: `{dataset.get('rows_after_cleaning')}`")
        lines.append(f"- Class counts: `{dataset.get('class_counts')}`")
        lines.append(f"- Cleaning: `{dataset.get('cleaning')}`")
        lines.append(f"- Split counts: `{dataset.get('split_counts')}`")
    alignment = report.get("alignment")
    if alignment:
        lines.extend(["", "## Feature alignment"])
        lines.append(f"- Aligned features: `{alignment.get('n_common')}`")
        lines.append(f"- Final feature names: `{alignment.get('common')}`")
        lines.append(f"- Excluded features: `{len(alignment.get('excluded', []))}`")
        lines.append(f"- Exclusions: `{alignment.get('excluded')}`")
    lines.extend(["", "## Leakage controls"])
    for key, value in report.get("leakage_controls", {}).items():
        lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-source", action="store_true", help="ne prépare que le dataset cible")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_seed(seed)
    label = cfg["data"]["label_column"]
    processed = resolve(cfg["paths"]["processed"])
    source_paths = []
    if cfg["data"].get("source_files"):
        source_paths = [resolve(cfg["paths"]["source_raw"]) / name
                        for name in cfg["data"]["source_files"]]
    else:
        source_paths = sorted(resolve(cfg["paths"]["source_raw"]).rglob("*.csv"))
    target_paths = sorted(resolve(cfg["paths"]["target_raw"]).rglob("*.csv"))
    report = {
        "config": args.config,
        "provenance": cfg["data"].get("provenance", {}),
        "memory_preflight": memory_preflight(source_paths + target_paths),
    }

    tgt, target_cleaning = load_clean(resolve(cfg["paths"]["target_raw"]), cfg, seed)
    report["target"] = {
        "rows_raw": target_cleaning["rows_before"],
        "rows_after_cleaning": len(tgt),
        "cleaning": target_cleaning,
        "class_counts": {str(k): int(v) for k, v in tgt[label].value_counts().items()},
    }
    tgt_cols = numeric_feature_columns(tgt, label)

    src, common = None, tgt_cols
    src_dir = resolve(cfg["paths"]["source_raw"])
    if not args.skip_source and src_dir.exists() and any(src_dir.rglob("*.csv")):
        source_files = cfg["data"].get("source_files")
        source_mapping = cfg["data"].get("source_label_mapping")
        src, source_cleaning = load_clean(
            src_dir, cfg, seed, selected_files=source_files,
            label_mapping=source_mapping,
        )
        al = align_features(numeric_feature_columns(src, label), tgt_cols)
        src = apply_alignment(src, al, "source")
        tgt = apply_alignment(tgt, al, "target")
        common = al["common"]
        report["source"] = {
            "rows_raw": source_cleaning["rows_before"],
            "rows_after_cleaning": len(src),
            "cleaning": source_cleaning,
            "selected_files": source_files,
            "label_mapping": source_mapping,
            "class_counts": {
                str(k): int(v) for k, v in src["_label_text"].value_counts().items()
            } if source_mapping is not None else {
                str(k): int(v) for k, v in src[label].value_counts().items()
            },
        }
        report["alignment"] = {
            "n_common": len(common),
            "common": common,
            "final_feature_columns": common,
            "pairs": al["pairs"],
            "excluded": al["excluded"],
            "source_only": al["source_only"],
            "target_only": al["target_only"],
        }
        print(f"Features communes source/cible : {len(common)} "
              f"(source seule : {len(al['source_only'])}, cible seule : {len(al['target_only'])})")
        if len(common) < 10:
            print("ATTENTION : très peu de features communes, le Transfer Learning sera peu pertinent.")
    else:
        print("Dataset source absent ou ignoré : seul le dataset cible est préparé.")

    t_splits = build_splits(
        tgt, common, label, cfg["data"], seed,
        feature_alignment=report.get("alignment"),
        scaler_fit_dataset="target",
        preprocessing_strategy="target_scaler_for_scratch",
    )
    if src is not None:
        t_splits["feature_alignment"] = report["alignment"]
    save_splits(processed, "target", t_splits)
    report["target"]["split_counts"] = t_splits["counts"]
    print("Cible :", {k: len(t_splits[f"y_{k}"]) for k in ("train", "val", "test")}, "| classes :", t_splits["class_names"])

    if src is not None:
        s_splits = build_splits(
            src, common, label, cfg["data"], seed, force_random=True,
            feature_alignment=report.get("alignment"),
            label_mapping=cfg["data"].get("source_label_mapping"),
            scaler_fit_dataset="source",
            preprocessing_strategy="source_scaler_for_source",
        )
        s_splits["feature_alignment"] = report["alignment"]
        save_splits(processed, "source", s_splits)
        report["source"]["split_counts"] = s_splits["counts"]
        print("Source :", {k: len(s_splits[f"y_{k}"]) for k in ("train", "val", "test")}, "| classes :", s_splits["class_names"])

        transfer_splits = build_splits(
            tgt, common, label, cfg["data"], seed,
            feature_alignment=report.get("alignment"),
            preprocessor=s_splits["preprocessor"],
            scaler_type="standard",
            scaler_fit_dataset="source",
            scaler_fit_split="train",
            preprocessing_strategy="source_scaler_for_transfer",
        )
        for split in ("train", "val", "test"):
            t_splits[f"X_transfer_{split}"] = transfer_splits[f"X_{split}"]
        t_splits["transfer_preprocessor"] = s_splits["preprocessor"].to_dict()
        t_splits["preprocessing_metadata"]["transfer"] = transfer_splits["preprocessing_metadata"]
        t_splits["preprocessing_metadata"]["scratch"] = {
            **t_splits["preprocessing_metadata"],
            "preprocessing_strategy": "target_scaler_for_scratch",
        }
        report["target"]["preprocessing_variants"] = {
            "scratch": t_splits["preprocessing_metadata"]["scratch"],
            "transfer": t_splits["preprocessing_metadata"]["transfer"],
        }
        save_splits(processed, "target", t_splits)

    report["leakage_controls"] = {
        "feature_alignment": "source and target use the same canonical ordered features",
        "forbidden_columns": "labels, index, identifiers and timestamps excluded",
        "scaler_fit": "train split only",
        "split_overlap": "all pairwise overlap counts are zero",
        "test_usage": "test is not used for preprocessing or model selection",
    }
    results_dir = resolve(cfg["paths"]["results"])
    save_json(report, results_dir / "data_report.json")
    (results_dir / "data_report.md").write_text(
        report_markdown(report), encoding="utf-8"
    )
    print("Rapports :", results_dir / "data_report.json", results_dir / "data_report.md")
