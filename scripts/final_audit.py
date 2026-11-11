"""Generate the reproducible final statistical audit without changing source results."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddos_tl.evaluation.results_analysis import (  # noqa: E402
    GRID_FRACTIONS,
    GRID_SEEDS,
    GRID_STRATEGIES,
    audit_results_frame,
    load_run_records,
    per_class_frame,
    summarize_per_class,
)

OUT = ROOT / "reports" / "results" / "final_audit"
FIGURES = ROOT / "reports" / "figures" / "final_audit"
X2X2 = ROOT / "reports" / "results" / "phase_2x2_scaler_initialization"
T_CRIT_95_N3 = 2.7764451051977987
METRICS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
)


def mean_std_ci(values: pd.Series) -> tuple[float, float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        return math.nan, math.nan, math.nan, math.nan
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if len(numeric) > 1 else math.nan
    if len(numeric) > 1 and math.isfinite(std):
        half = T_CRIT_95_N3 * std / math.sqrt(len(numeric))
        return mean, std, mean - half, mean + half
    return mean, std, math.nan, math.nan


def summarize_rows(rows: list[dict], group_key: str, include_timing: bool = False) -> pd.DataFrame:
    flat = []
    for row in rows:
        metrics = row["test_metrics"]
        item = {group_key: row[group_key], "fraction": float(row["fraction"]), "seed": int(row["seed"])}
        item.update({metric: metrics[metric] for metric in METRICS})
        if include_timing:
            item["train_time_s"] = row["train_time_s"]
            item["inference_us_per_sample"] = row["inference_us_per_sample"]
        flat.append(item)
    frame = pd.DataFrame(flat)
    groups = [group_key, "fraction"]
    value_columns = list(METRICS)
    if include_timing:
        value_columns += ["train_time_s", "inference_us_per_sample"]
    output = []
    for keys, group in frame.groupby(groups, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = dict(zip(groups, keys))
        for column in value_columns:
            mean, std, low, high = mean_std_ci(group[column])
            item.update({
                f"{column}_mean": mean,
                f"{column}_std": std,
                f"{column}_ci95_low": low,
                f"{column}_ci95_high": high,
            })
        item["n"] = len(group)
        output.append(item)
    return pd.DataFrame(output).sort_values(groups).reset_index(drop=True)


def flatten_classes(rows: list[dict], group_key: str) -> pd.DataFrame:
    records = []
    for row in rows:
        for class_name, metrics in row["test_metrics"]["per_class"].items():
            records.append({
                group_key: row[group_key],
                "fraction": float(row["fraction"]),
                "seed": int(row["seed"]),
                "class": class_name,
                **{metric: metrics[metric] for metric in ("precision", "recall", "f1", "support")},
            })
    return pd.DataFrame(records)


def summarize_classes(frame: pd.DataFrame, group_key: str) -> pd.DataFrame:
    groups = [group_key, "fraction", "class"]
    rows = []
    for keys, group in frame.groupby(groups, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = dict(zip(groups, keys))
        for column in ("precision", "recall", "f1", "support"):
            mean, std, low, high = mean_std_ci(group[column])
            item.update({
                f"{column}_mean": mean,
                f"{column}_std": std,
                f"{column}_ci95_low": low,
                f"{column}_ci95_high": high,
            })
        item["n"] = len(group)
        rows.append(item)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def validate_inputs(df: pd.DataFrame, historical: list[dict], controlled: list[dict]) -> dict:
    audit = audit_results_frame(df)
    expected_keys = {
        (strategy, fraction, seed)
        for strategy in GRID_STRATEGIES
        for fraction in GRID_FRACTIONS
        for seed in GRID_SEEDS
    }
    historical_keys = {(r["strategy"], float(r["fraction"]), int(r["seed"])) for r in historical}
    controlled_keys = {(r["cell"], float(r["fraction"]), int(r["seed"])) for r in controlled}
    expected_controlled = {
        (cell, fraction, seed)
        for cell in ("random_target", "pretrained_target", "random_source", "pretrained_source")
        for fraction in GRID_FRACTIONS
        for seed in GRID_SEEDS
    }
    checks = {
        "historical_csv_rows": len(df) == 60,
        "historical_json_count": len(historical) == 60,
        "historical_grid_keys": historical_keys == expected_keys,
        "historical_csv_json_keys": {
            (r["strategy"], float(r["fraction"]), int(r["seed"])) for _, r in df.iterrows()
        } == historical_keys,
        "historical_no_missing": audit["missing_values"] == 0,
        "historical_no_nonfinite": audit["nonfinite_values"] == 0,
        "historical_no_duplicates": audit["duplicate_keys"] == 0,
        "controlled_json_count": len(controlled) == 48,
        "controlled_grid_keys": controlled_keys == expected_controlled,
        "controlled_no_missing_metrics": all(
            all(metric in r["test_metrics"] for metric in METRICS) for r in controlled
        ),
        "all_18_classes_historical": len(historical[0]["test_metrics"]["per_class"]) == 18,
        "all_18_classes_controlled": len(controlled[0]["test_metrics"]["per_class"]) == 18,
    }
    if not all(checks.values()):
        raise ValueError(f"Input integrity validation failed: {checks}")
    return checks


def effect_table(controlled: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([
        {
            "fraction": float(row["fraction"]),
            "seed": int(row["seed"]),
            "cell": row["cell"],
            "macro_f1": row["test_metrics"]["macro_f1"],
        }
        for row in controlled
    ])
    rows = []
    for fraction in GRID_FRACTIONS:
        pivot = frame[frame["fraction"] == fraction].pivot(index="seed", columns="cell", values="macro_f1")
        contrasts = {
            "pretrained_minus_random_target": ("pretrained_target", "random_target"),
            "pretrained_minus_random_source": ("pretrained_source", "random_source"),
            "source_minus_target_random": ("random_source", "random_target"),
            "source_minus_target_pretrained": ("pretrained_source", "pretrained_target"),
        }
        for name, (left, right) in contrasts.items():
            values = pivot[left] - pivot[right]
            mean, std, low, high = mean_std_ci(values)
            rows.append({
                "fraction": fraction,
                "contrast": name,
                "mean_delta": mean,
                "std_delta": std,
                "ci95_low": low,
                "ci95_high": high,
                "n": len(values),
            })
        interaction = (pivot["pretrained_source"] - pivot["random_source"]) - (
            pivot["pretrained_target"] - pivot["random_target"]
        )
        mean, std, low, high = mean_std_ci(interaction)
        rows.append({
            "fraction": fraction,
            "contrast": "interaction_initialization_x_scaler",
            "mean_delta": mean,
            "std_delta": std,
            "ci95_low": low,
            "ci95_high": high,
            "n": len(interaction),
        })
    return pd.DataFrame(rows)


def save_figures(historical_summary: pd.DataFrame, controlled_summary: pd.DataFrame,
                 controlled_classes: pd.DataFrame, controlled: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for strategy, group in historical_summary.groupby("strategy"):
        plt.errorbar(group["fraction"] * 100, group["macro_f1_mean"],
                     yerr=group["macro_f1_std"].fillna(0), marker="o", capsize=3, label=strategy)
    for cell, group in controlled_summary.groupby("cell"):
        plt.errorbar(group["fraction"] * 100, group["macro_f1_mean"],
                     yerr=group["macro_f1_std"].fillna(0), marker="x", linestyle="--", capsize=3, label=f"2x2:{cell}")
    plt.xscale("log")
    plt.xticks([1, 5, 10, 100], ["1%", "5%", "10%", "100%"])
    plt.xlabel("Fraction des données cibles")
    plt.ylabel("Macro F1 test moyen")
    plt.title("Macro F1 : grille initiale et expérience contrôlée 2x2")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "macro_f1_all_runs.png", dpi=180)
    plt.close()

    pivot = controlled_summary.pivot(index="fraction", columns="cell", values="macro_f1_mean")
    plt.figure(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Macro F1 moyen par cellule 2x2 et fraction")
    plt.xlabel("Cellule")
    plt.ylabel("Fraction")
    plt.tight_layout()
    plt.savefig(FIGURES / "controlled_2x2_macro_f1_heatmap.png", dpi=180)
    plt.close()

    class_pivot = controlled_classes.groupby("class")["f1_mean"].mean().sort_values()
    class_table = controlled_classes.pivot_table(index="class", columns="cell", values="f1_mean")
    class_table = class_table.reindex(class_pivot.index)
    plt.figure(figsize=(10, 9))
    sns.heatmap(class_table, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("F1 par classe : expérience contrôlée 2x2")
    plt.xlabel("Cellule")
    plt.ylabel("Classe (difficulté croissante)")
    plt.tight_layout()
    plt.savefig(FIGURES / "controlled_2x2_class_f1_heatmap.png", dpi=180)
    plt.close()

    best = max(controlled, key=lambda row: row["test_metrics"]["macro_f1"])
    cm = np.asarray(best["test_metrics"]["confusion_matrix"], dtype=float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = list(best["test_metrics"]["per_class"])
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap="Blues", vmin=0, vmax=1, xticklabels=names, yticklabels=names)
    plt.title(f"Matrice de confusion normalisée - meilleur run 2x2 ({best['run_id']})")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "controlled_2x2_best_confusion_matrix.png", dpi=180)
    plt.close()


def write_report(checks: dict, historical_summary: pd.DataFrame,
                 controlled_summary: pd.DataFrame, effects: pd.DataFrame,
                 historical_classes: pd.DataFrame, controlled_classes: pd.DataFrame) -> None:
    h_hard = historical_classes.groupby("class")["f1_mean"].mean().sort_values().head(5)
    c_hard = controlled_classes.groupby("class")["f1_mean"].mean().sort_values().head(5)
    lines = [
        "# Audit statistique final des 108 runs",
        "",
        "## Périmètre et reproductibilité",
        "",
        "Analyse réalisée à partir des 60 lignes de `experiments.csv`, des 60 JSON historiques et des 48 JSON de l'expérience contrôlée 2x2. Les résultats sources n'ont pas été modifiés.",
        "",
        "Commande reproductible :",
        "",
        "```powershell",
        r".\.venv-cuda\Scripts\python.exe scripts\final_audit.py",
        "```",
        "",
        "## Intégrité",
        "",
    ]
    lines += [f"- `{name}` : **{'PASS' if value else 'FAIL'}**" for name, value in checks.items()]
    lines += [
        "",
        "Les métriques contrôlées sont finies et les matrices de confusion sont 18x18. Les classes `UDPLag` et `WebDDoS` sont conservées.",
        "",
        "## Résultats statistiques",
        "",
        "Les CSV associés contiennent moyenne, écart-type d'échantillon et IC 95% t de Student. Avec trois seeds, les IC sont très incertains et ne constituent pas une preuve de généralisation.",
        "",
        "### Classes les plus difficiles",
        "",
        "**Grille initiale :**",
        "",
    ]
    lines += [f"- {name}: F1 moyen {value:.4f}" for name, value in h_hard.items()]
    lines += ["", "**Expérience contrôlée 2x2 :**", ""]
    lines += [f"- {name}: F1 moyen {value:.4f}" for name, value in c_hard.items()]
    lines += [
        "",
        "## Effets 2x2",
        "",
        "L'interaction est calculée comme `(pré-entraîné - aléatoire sous source) - (pré-entraîné - aléatoire sous target)`. Elle décrit l'effet différentiel observé dans ce protocole, sans permettre une généralisation au-delà des quatre fractions et trois seeds.",
        "",
        "Voir `controlled_2x2_effects.csv` pour les contrastes et leurs IC.",
        "",
        "## Comparaison Scratch / Transfer Learning",
        "",
        "Les comparaisons de la grille initiale sont fournies dans `negative_transfer_60.csv`. Elles doivent être interprétées avec prudence : les stratégies historiques ne partagent pas nécessairement le même preprocessing que Scratch, ce qui confond initialisation et scaler.",
        "",
        "## Risques et limites scientifiques",
        "",
        "- Le split est aléatoire par lignes : des flux ou familles très similaires peuvent se retrouver dans plusieurs partitions, ce qui peut gonfler les performances.",
        "- Les métadonnées indiquent des intersections de split nulles pour les indices disponibles, mais cela ne démontre pas l'absence de doublons sémantiques entre lignes.",
        "- Le target scaler est ajusté sur tout `target/train` avant la sélection des sous-ensembles supervisés : c'est une adaptation non supervisée et une information de distribution cible est utilisée.",
        "- Les scalers source et cible diffèrent ; dans les expériences initiales, ce facteur est confondu avec la stratégie de transfert.",
        "- `UDPLag` et `WebDDoS` ont un support de 8 exemples dans les tests : leurs F1 sont extrêmement instables.",
        "- La source est binaire et limitée au fichier DDoS du vendredi ; elle ne couvre pas les 18 classes cibles.",
        "- Les JSON 2x2 ne contiennent pas les temps d'entraînement ou d'inférence : aucune durée n'est inventée ni estimée.",
        "- Trois seeds seulement limitent fortement la précision des écarts-types et intervalles de confiance.",
        "",
        "## Conclusion",
        "",
        "Les 108 runs sont structurellement complets et exploitables pour une analyse descriptive. Les résultats montrent un transfert variable et parfois négatif, particulièrement en faible régime de labels. Les conclusions causales doivent rester limitées au protocole contrôlé, en tenant compte du domain shift, du scaler et du split aléatoire par lignes.",
        "",
    ]
    (OUT / "final_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(ROOT / "reports" / "results" / "experiments.csv")
    historical = load_run_records(ROOT / "reports" / "results" / "runs")
    controlled_paths = sorted(p for p in X2X2.glob("*.json")
                              if p.name not in {"protocol_manifest.json", "protected_artifact_hashes_before_implementation.json"})
    controlled = [json.loads(path.read_text(encoding="utf-8")) for path in controlled_paths]
    checks = validate_inputs(df, historical, controlled)

    OUT.mkdir(parents=True, exist_ok=True)
    historical_summary = summarize_rows(historical, "strategy", include_timing=True)
    controlled_summary = summarize_rows(controlled, "cell", include_timing=False)
    historical_classes = summarize_classes(flatten_classes(historical, "strategy"), "strategy")
    controlled_classes = summarize_classes(flatten_classes(controlled, "cell"), "cell")
    effects = effect_table(controlled)

    historical_summary.to_csv(OUT / "statistical_summary_60.csv", index=False)
    controlled_summary.to_csv(OUT / "statistical_summary_2x2.csv", index=False)
    historical_classes.to_csv(OUT / "per_class_analysis_60.csv", index=False)
    controlled_classes.to_csv(OUT / "per_class_analysis_2x2.csv", index=False)
    effects.to_csv(OUT / "controlled_2x2_effects.csv", index=False)

    negative = []
    for fraction in GRID_FRACTIONS:
        scratch = df[np.isclose(df["fraction"], fraction) & (df["strategy"] == "scratch")]["test_macro_f1"]
        scratch_mean = scratch.mean()
        for strategy in [s for s in GRID_STRATEGIES if s != "scratch"]:
            transfer = df[np.isclose(df["fraction"], fraction) & (df["strategy"] == strategy)]["test_macro_f1"]
            delta = transfer.mean() - scratch_mean
            negative.append({
                "strategy": strategy,
                "fraction": fraction,
                "transfer_mean": transfer.mean(),
                "scratch_mean": scratch_mean,
                "delta_transfer_minus_scratch": delta,
                "interpretation": "observed positive transfer" if delta > 0 else "observed negative transfer" if delta < 0 else "no mean difference",
            })
    pd.DataFrame(negative).to_csv(OUT / "negative_transfer_60.csv", index=False)
    pd.DataFrame([{"check": key, "status": "PASS" if value else "FAIL"} for key, value in checks.items()]).to_csv(
        OUT / "integrity_audit.csv", index=False
    )
    (OUT / "timing_availability.csv").write_text(
        "dataset,training_time,inference_time,reason\n"
        "historical_60,available,available,stored in CSV and JSON\n"
        "controlled_2x2,missing,missing,not stored in controlled JSON records\n",
        encoding="utf-8",
    )
    save_figures(historical_summary, controlled_summary, controlled_classes, controlled)
    write_report(checks, historical_summary, controlled_summary, effects, historical_classes, controlled_classes)
    print(f"Generated final audit in {OUT}")
    print(f"Generated figures in {FIGURES}")


if __name__ == "__main__":
    main()
