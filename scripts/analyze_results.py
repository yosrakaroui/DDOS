"""Generate non-destructive scientific analysis from the completed experiment grid."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.evaluation.plots import plot_confusion_matrix
from ddos_tl.evaluation.results_analysis import (
    GRID_FRACTIONS,
    GRID_SEEDS,
    GRID_STRATEGIES,
    METRICS,
    audit_results_frame,
    load_run_records,
    negative_transfer_frame,
    per_class_frame,
    summarize_per_class,
    summarize_statistics,
)
from ddos_tl.utils.io import load_json


def _save_line_plot(summary: pd.DataFrame, metric: str, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, group in summary.groupby("strategy"):
        group = group.sort_values("fraction")
        ax.errorbar(
            group["fraction"] * 100,
            group[f"{metric}_mean"],
            yerr=group[f"{metric}_std"].fillna(0),
            marker="o",
            capsize=3,
            label=strategy,
        )
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 100], ["1%", "5%", "10%", "100%"])
    ax.set_xlabel("Fraction des labels cibles")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_metric_comparison(summary: pd.DataFrame, path: Path) -> None:
    long = summary.melt(
        id_vars=["strategy", "fraction"],
        value_vars=["test_accuracy_mean", "test_macro_f1_mean"],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map({
        "test_accuracy_mean": "Accuracy",
        "test_macro_f1_mean": "Macro F1",
    })
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long, x="fraction", y="value", hue="strategy", errorbar=None, ax=ax)
    ax.set_xlabel("Fraction des labels cibles")
    ax.set_ylabel("Moyenne")
    ax.set_title("Accuracy et Macro F1 moyennes")
    ax.legend(title="Stratégie", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_runtime_plot(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, group in summary.groupby("strategy"):
        group = group.sort_values("fraction")
        ax.plot(group["fraction"] * 100, group["train_time_s_mean"], marker="o", label=strategy)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 100], ["1%", "5%", "10%", "100%"])
    ax.set_xlabel("Fraction des labels cibles")
    ax.set_ylabel("Temps d'entraînement moyen (s)")
    ax.set_title("Temps d'entraînement selon la fraction")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_class_heatmap(class_summary: pd.DataFrame, path: Path) -> None:
    heat = class_summary.groupby("class")["f1_mean"].mean().sort_values()
    table = class_summary.pivot_table(index="class", columns="strategy", values="f1_mean")
    table = table.reindex(heat.index)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("F1 moyenne par classe, stratégie et fraction")
    ax.set_xlabel("Stratégie")
    ax.set_ylabel("Classe (de la plus difficile à la plus facile)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _best_run(records: list[dict], strategy: str, fraction: float) -> dict:
    candidates = [
        record for record in records
        if record["strategy"] == strategy and np.isclose(record["fraction"], fraction)
    ]
    return max(candidates, key=lambda record: record["val_macro_f1"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="reports/results")
    parser.add_argument("--figures", default="reports/figures")
    args = parser.parse_args()
    results = Path(args.results)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)

    csv_path = results / "experiments.csv"
    df = pd.read_csv(csv_path)
    audit = audit_results_frame(df)
    if not audit["row_count_ok"] or audit["missing_values"] or audit["nonfinite_values"] or audit["duplicate_keys"]:
        raise ValueError(f"Invalid experiment grid: {audit}")
    if audit["unexpected_strategies"] or audit["unexpected_fractions"] or audit["unexpected_seeds"]:
        raise ValueError(f"Unexpected grid values: {audit}")

    records = load_run_records(results / "runs")
    expected_keys = {
        (strategy, fraction, seed)
        for strategy in GRID_STRATEGIES for fraction in GRID_FRACTIONS for seed in GRID_SEEDS
    }
    record_keys = {(r["strategy"], float(r["fraction"]), int(r["seed"])) for r in records}
    if record_keys != expected_keys:
        raise ValueError(f"Detailed run JSON keys do not match grid: {len(record_keys)}")

    summary = summarize_statistics(df)
    summary.to_csv(results / "statistical_summary.csv", index=False)
    classes = per_class_frame(records)
    class_summary = summarize_per_class(classes)
    class_summary.to_csv(results / "per_class_analysis.csv", index=False)
    negative = negative_transfer_frame(df)
    negative.to_csv(results / "negative_transfer_analysis.csv", index=False)

    _save_line_plot(summary, "test_macro_f1", figures / "complete_analysis_macro_f1.png",
                    "Macro F1 selon la quantité de données", "Macro F1 test")
    _save_metric_comparison(summary, figures / "complete_analysis_accuracy_macro_f1.png")
    _save_runtime_plot(summary, figures / "complete_analysis_training_time.png")
    _save_class_heatmap(class_summary, figures / "complete_analysis_class_f1_heatmap.png")

    for strategy in GRID_STRATEGIES:
        run = _best_run(records, strategy, 1.0)
        plot_confusion_matrix(
            run["test_metrics"]["confusion_matrix"],
            list(run["test_metrics"]["per_class"]),
            figures / f"complete_analysis_confusion_{strategy}_100pct.png",
        )

    hardest = class_summary.groupby("class")["f1_mean"].mean().sort_values().head(5)
    transfer_negative = negative[negative["delta_transfer_minus_scratch"] < 0]
    macro_table = summary.pivot(index="fraction", columns="strategy", values="test_macro_f1_mean")
    macro_lines = [
        "| Fraction | " + " | ".join(GRID_STRATEGIES) + " |",
        "|" + "|".join(["---"] * (len(GRID_STRATEGIES) + 1)) + "|",
    ]
    for fraction, row in macro_table.sort_index().iterrows():
        macro_lines.append(
            "| {:.0%} | {} |".format(
                fraction,
                " | ".join(f"{row.get(strategy, float('nan')):.4f}" for strategy in GRID_STRATEGIES),
            )
        )
    hardest_lines = "".join(f"- {name}: F1 moyen {value:.4f}\n" for name, value in hardest.items())
    macro_table_text = "\n".join(macro_lines)
    report = f"""# Analyse scientifique des 60 expériences

## Audit

- Expériences CSV vérifiées : **{len(df)}/60**.
- Fichiers JSON détaillés vérifiés : **{len(records)}/60**.
- Valeurs manquantes : **{audit['missing_values']}**.
- Valeurs non finies : **{audit['nonfinite_values']}**.
- Doublons de clé : **{audit['duplicate_keys']}**.
- Stratégies : `{', '.join(GRID_STRATEGIES)}`.
- Fractions : `{list(GRID_FRACTIONS)}`.
- Seeds : `{list(GRID_SEEDS)}`.

## Résultats agrégés

Le fichier `statistical_summary.csv` fournit, pour chaque stratégie et fraction,
les moyennes et écarts-types (écart-type d'échantillon sur les trois seeds) de
l'Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1 et des temps.
La métrique primaire est la Macro F1.

### Macro F1 test moyen

{macro_table_text}

## Negative transfer observé

`negative_transfer_analysis.csv` compare chaque stratégie transférée à Scratch
à fraction identique. Un delta négatif signifie que la moyenne Macro F1 de la
stratégie transférée est inférieure à Scratch dans les résultats observés.
{len(transfer_negative)} comparaisons stratégie-fraction ont un delta négatif.
Cela décrit les données observées et ne constitue pas une preuve de causalité.

## Analyse par classe

`per_class_analysis.csv` conserve les 18 classes, leur support, et les moyennes
et écarts-types de Precision, Recall et F1. Les cinq classes au F1 moyen le plus
faible sur toutes les conditions sont :

{hardest_lines}

Les classes rares `UDPLag` et `WebDDoS` sont conservées. Leur support test est
faible dans les artefacts disponibles ; leurs métriques sont donc très
instables et doivent être interprétées avec prudence.

## Limites scientifiques

Scratch utilise le scaler cible tandis que les stratégies de transfert utilisent
le scaler source. L'écart observé combine donc initialisation pré-entraînée,
stratégie d'adaptation et représentation numérique. Un écart négatif ne peut
pas être attribué automatiquement au pré-entraînement. Le domain shift, le
faible support des classes rares et le nombre limité de seeds limitent aussi la
généralisation.

Les figures associées se trouvent dans `reports/figures/complete_analysis/`.
"""
    (results / "complete_analysis.md").write_text(report, encoding="utf-8")
    (results / "limitations.md").write_text(
        """# Limitations de l'analyse

1. Le protocole utilise trois seeds, ce qui limite la précision de l'estimation
   de la variabilité.
2. Scratch et Transfer n'utilisent pas le même scaler : cible pour Scratch,
   source pour Transfer. Cette différence est un facteur de confusion.
3. Le scaler cible est ajusté sur toutes les features de `target/train` avant
   la sélection des labels low-data ; il s'agit d'une adaptation cible
   non-supervisée, pas d'un protocole strictement limité aux lignes étiquetées.
4. Les classes `UDPLag` et `WebDDoS` ont un support faible, ce qui rend leur
   F1 très variable.
5. Les résultats établissent des différences observées entre stratégies, mais
   ne démontrent pas à eux seuls une causalité du pré-entraînement.
6. La matrice 2x2 scaler/initialisation proposée en phase F.4 n'est pas incluse
   dans ces 60 expériences.
""", encoding="utf-8")
    print(f"Analysis written to {results}; figures written to {figures}")


if __name__ == "__main__":
    main()
