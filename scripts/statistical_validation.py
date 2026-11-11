"""Validate final-audit statistics without changing raw results or old reports."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "results" / "final_audit" / "statistical_validation"
HISTORICAL = ROOT / "reports" / "results" / "runs"
CONTROLLED = ROOT / "reports" / "results" / "phase_2x2_scaler_initialization"
FRACTIONS = (0.01, 0.05, 0.10, 1.0)
SEEDS = (0, 1, 2)
CELLS = ("random_target", "pretrained_target", "random_source", "pretrained_source")
T_CRIT_95_DF2 = 4.302652729911275


def ci95(values: list[float]) -> tuple[float, float, float, float]:
    """Return mean, sample SD and 95% t interval for exactly three seeds."""
    array = np.asarray(values, dtype=float)
    if len(array) != 3 or not np.isfinite(array).all():
        raise ValueError("A confidence interval requires three finite paired observations")
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    half_width = T_CRIT_95_DF2 * std / math.sqrt(len(array))
    return mean, std, mean - half_width, mean + half_width


def load_json_records(directory: Path) -> list[dict]:
    excluded = {"protocol_manifest.json", "protected_artifact_hashes_before_implementation.json"}
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json")) if path.name not in excluded]


def validate_records(historical: list[dict], controlled: list[dict]) -> None:
    if len(historical) != 60 or len(controlled) != 48:
        raise ValueError("Expected exactly 60 historical and 48 controlled records")
    for record in historical + controlled:
        metrics = record["test_metrics"]
        if not math.isfinite(float(metrics["macro_f1"])):
            raise ValueError(f"Non-finite Macro F1 in {record.get('run_id', record.get('strategy'))}")


def summarize_groups(records: list[dict], key: str) -> pd.DataFrame:
    rows = []
    for group in sorted({record[key] for record in records}):
        for fraction in FRACTIONS:
            values = [
                float(record["test_metrics"]["macro_f1"])
                for record in records
                if record[key] == group and float(record["fraction"]) == fraction
            ]
            mean, std, low, high = ci95(values)
            rows.append({
                key: group, "fraction": fraction, "n": len(values),
                "macro_f1_mean": mean, "macro_f1_std": std,
                "ci95_low": low, "ci95_high": high,
                "ci_method": "t_critical=4.302653, df=2",
            })
    return pd.DataFrame(rows)


def paired_effects(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([
        {
            "fraction": float(record["fraction"]),
            "seed": int(record["seed"]),
            "cell": record["cell"],
            "macro_f1": float(record["test_metrics"]["macro_f1"]),
        }
        for record in records
    ])
    rows = []
    contrasts = {
        "pretrained_minus_random_target": ("pretrained_target", "random_target"),
        "pretrained_minus_random_source": ("pretrained_source", "random_source"),
        "source_minus_target_random": ("random_source", "random_target"),
        "source_minus_target_pretrained": ("pretrained_source", "pretrained_target"),
    }
    for fraction in FRACTIONS:
        subset = frame[frame["fraction"] == fraction].pivot(
            index="seed", columns="cell", values="macro_f1"
        ).reindex(index=SEEDS, columns=CELLS)
        if subset.isna().any().any():
            raise ValueError(f"Missing paired seed in fraction {fraction}")
        for name, (left, right) in contrasts.items():
            differences = (subset[left] - subset[right]).tolist()
            mean, std, low, high = ci95(differences)
            rows.append({
                "fraction": fraction, "contrast": name,
                "mean_delta": mean, "std_delta": std,
                "ci95_low": low, "ci95_high": high, "n": 3,
                "paired": True,
                "evidence": "descriptive_only",
                "interpretation": (
                    "CI excludes zero; exploratory only"
                    if low > 0 or high < 0 else "CI includes zero"
                ),
            })
        interaction = (
            subset["pretrained_source"] - subset["random_source"]
            - subset["pretrained_target"] + subset["random_target"]
        ).tolist()
        mean, std, low, high = ci95(interaction)
        rows.append({
            "fraction": fraction,
            "contrast": "interaction_initialization_x_preprocessing",
            "mean_delta": mean, "std_delta": std,
            "ci95_low": low, "ci95_high": high, "n": 3,
            "paired": True,
            "evidence": "descriptive_only",
            "interpretation": (
                "CI excludes zero; exploratory only"
                if low > 0 or high < 0 else "CI includes zero"
            ),
        })
    return pd.DataFrame(rows)


def write_report(historical_summary: pd.DataFrame, controlled_summary: pd.DataFrame,
                 effects: pd.DataFrame) -> None:
    old_t = 2.7764451051977987
    corrected = effects[effects["contrast"] == "pretrained_minus_random_target"]
    lines = [
        "# Validation statistique complémentaire",
        "",
        "## Périmètre",
        "",
        "Cette validation relit les 60 runs historiques et les 48 runs contrôlés 2x2. Aucun résultat brut, dataset, checkpoint ou ancien rapport n'a été modifié.",
        "",
        "## Audit du calcul existant",
        "",
        "`scripts/final_audit.py` définissait `T_CRIT_95_N3 = 2.776445`. Cette valeur est la valeur t bilatérale à 95% pour `df=4`, pas pour trois seeds. Pour un intervalle t bilatéral à 95% avec `n=3`, `df=n-1=2` et `t critique=4.302653`.",
        "",
        "Correction appliquée uniquement dans cette validation complémentaire :",
        "",
        "- moyenne arithmétique sur les trois seeds ;",
        "- écart-type d'échantillon (`ddof=1`) ;",
        "- IC 95% : `moyenne ± 4.302653 × écart-type / sqrt(3)` ;",
        "- différences calculées de façon appariée par seed pour les cellules 2x2.",
        "",
        "Les anciens fichiers restent volontairement inchangés et leurs IC doivent être considérés comme à corriger lors d'une future régénération explicitement autorisée.",
        "",
        "## Vérification des moyennes et écarts-types",
        "",
        "Les tableaux `summary_60_corrected.csv` et `summary_2x2_corrected.csv` contiennent les moyennes, écarts-types et IC corrigés. Les écarts-types utilisent l'estimateur d'échantillon.",
        "",
        "## Effets appariés 2x2",
        "",
        "Chaque contraste compare les mêmes trois seeds dans les deux cellules. L'interaction est : `(pretrained_source - random_source) - (pretrained_target - random_target)`.",
        "",
        "Les contrastes sont des estimations descriptives. Un IC qui exclut zéro est signalé comme indice exploratoire, pas comme preuve confirmatoire.",
        "",
        "## Pertinence des tests avec trois seeds",
        "",
        "Un test t apparié est mathématiquement calculable avec trois paires, mais il est extrêmement peu puissant et très sensible à une seule seed. Les hypothèses de normalité des différences ne peuvent pas être évaluées sérieusement avec `n=3`. Les seeds ne constituent pas nécessairement un échantillonnage aléatoire d'une population de protocoles.",
        "",
        "Par conséquent :",
        "",
        "- aucune différence n'est déclarée statistiquement significative ;",
        "- aucun test de normalité n'est interprété ;",
        "- les intervalles sont utilisés pour quantifier l'incertitude, non pour établir une preuve causale ;",
        "- une conclusion confirmatoire nécessiterait davantage de seeds, un protocole pré-enregistré et idéalement une analyse de puissance.",
        "",
        "## Interprétation scientifique",
        "",
        "- L'initialisation et le preprocessing ont des effets variables selon la fraction de données.",
        "- L'interaction peut changer de signe selon la fraction ; elle ne doit pas être résumée par un effet global unique.",
        "- Les contrastes 2x2 sont mieux identifiés causalement que les comparaisons historiques, car les cellules contrôlées partagent les seeds, sous-ensembles et hyperparamètres prévus.",
        "- Les comparaisons des 60 runs restent confondues par le protocole de preprocessing historique et ne doivent pas être interprétées comme un effet pur de l'initialisation.",
        "",
        "## Fichiers",
        "",
        "- `summary_60_corrected.csv`",
        "- `summary_2x2_corrected.csv`",
        "- `paired_effects_2x2.csv`",
        "- `validation_checks.csv`",
        "",
        "Les valeurs corrigées sont reproductibles avec `scripts/statistical_validation.py`.",
    ]
    (OUT / "statistical_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    historical = load_json_records(HISTORICAL)
    controlled = load_json_records(CONTROLLED)
    validate_records(historical, controlled)
    OUT.mkdir(parents=True, exist_ok=True)
    historical_summary = summarize_groups(historical, "strategy")
    controlled_summary = summarize_groups(controlled, "cell")
    effects = paired_effects(controlled)
    historical_summary.to_csv(OUT / "summary_60_corrected.csv", index=False)
    controlled_summary.to_csv(OUT / "summary_2x2_corrected.csv", index=False)
    effects.to_csv(OUT / "paired_effects_2x2.csv", index=False)
    pd.DataFrame([
        {"check": "historical_records", "value": len(historical), "status": "PASS"},
        {"check": "controlled_records", "value": len(controlled), "status": "PASS"},
        {"check": "t_critical_95_df2", "value": T_CRIT_95_DF2, "status": "PASS"},
        {"check": "paired_seed_alignment", "value": "3 seeds per cell/fraction", "status": "PASS"},
        {"check": "inferential_claims", "value": "none; descriptive_only", "status": "PASS"},
    ]).to_csv(OUT / "validation_checks.csv", index=False)
    write_report(historical_summary, controlled_summary, effects)
    print(f"Generated corrected statistical validation in {OUT}")


if __name__ == "__main__":
    main()
