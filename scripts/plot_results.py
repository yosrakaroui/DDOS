"""Phase 10 : figures et tableau récapitulatif à partir de experiments.csv."""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config, resolve
from ddos_tl.evaluation.plots import plot_confusion_matrix
from ddos_tl.utils.io import load_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    results, figures = resolve(cfg["paths"]["results"]), resolve(cfg["paths"]["figures"])
    figures.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results / "experiments.csv")

    agg = df.groupby(["strategy", "fraction"])["test_macro_f1"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, g in agg.groupby("strategy"):
        ax.errorbar(g["fraction"] * 100, g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=3, label=strategy)
    ax.set_xscale("log")
    ax.set_xlabel("% des données d'entraînement cibles (échelle log)")
    ax.set_ylabel("Macro F1 (test)")
    ax.set_title("Transfer Learning vs entraînement from scratch")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "lowdata_macro_f1.png", dpi=150)

    # Tableau récapitulatif (moyenne ± écart-type) au format Markdown
    metric_columns = [
        ("test_accuracy", "Accuracy"),
        ("test_macro_precision", "Macro Precision"),
        ("test_macro_recall", "Macro Recall"),
        ("test_macro_f1", "Macro F1"),
        ("test_weighted_f1", "Weighted F1"),
    ]
    headers = " | ".join(["Stratégie", "Fraction"] + [name for _, name in metric_columns])
    lines = [f"| {headers} |", "|" + "|".join(["---"] * (2 + len(metric_columns))) + "|"]
    for (s, f), g in df.groupby(["strategy", "fraction"]):
        values = [
            f"{g[column].mean():.4f} ± {g[column].std(ddof=0):.4f}"
            for column, _ in metric_columns
        ]
        lines.append("| " + " | ".join([str(s), str(f)] + values) + " |")
    lines.extend(["", "## Negative transfer analysis", ""])
    for fraction, group in df.groupby("fraction"):
        scratch = group[group["strategy"] == "scratch"]["test_macro_f1"].mean()
        transfer = group[group["strategy"] != "scratch"]["test_macro_f1"].mean()
        if pd.notna(scratch) and pd.notna(transfer):
            status = "negative transfer" if transfer < scratch else "no negative transfer"
            lines.append(
                f"- Fraction `{fraction}`: transfer mean `{transfer:.4f}` vs "
                f"scratch `{scratch:.4f}` — **{status}**."
            )
    (results / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    run_records = []
    for path in sorted((results / "runs").glob("*.json")):
        run = load_json(path)
        for class_name, values in run.get("test_metrics", {}).get("per_class", {}).items():
            run_records.append({
                "strategy": run.get("strategy"),
                "fraction": run.get("fraction"),
                "seed": run.get("seed"),
                "class": class_name,
                **{metric: values[metric] for metric in ("precision", "recall", "f1", "support")},
            })
    if run_records:
        per_class = pd.DataFrame(run_records)
        per_class.to_csv(results / "per_class_metrics.csv", index=False)
        per_class.groupby(["strategy", "fraction", "class"])[
            ["precision", "recall", "f1"]
        ].agg(["mean", "std"]).reset_index().to_csv(
            results / "per_class_metrics_summary.csv", index=False
        )

    # Select the finetune run with the highest validation Macro F1.
    runs = sorted((results / "runs").glob("finetune_f1.0_s*.json"))
    if runs:
        run = max(runs, key=lambda path: load_json(path).get("val_macro_f1", -1.0))
        class_names = list(run["test_metrics"]["per_class"].keys())
        plot_confusion_matrix(run["test_metrics"]["confusion_matrix"], class_names, figures / "confusion_finetune.png")
    print("Figures :", figures, "| Tableau :", results / "summary.md")
