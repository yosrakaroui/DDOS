# Chronologie du projet

Les statuts ci-dessous sont fondés sur les scripts, métadonnées et rapports
présents dans le repository. Une commande est indiquée comme reproductible
mais n'est pas exécutée dans le cadre de cette documentation.

| Étape | Objectif et méthode | Fichiers principaux | Résultat / statut |
|---|---|---|---|
| Initialisation | Structurer un package Python, une configuration YAML et des tests | `pyproject.toml`, `configs/`, `src/`, `tests/` | Terminé |
| Environnement | Prévoir venv CPU/GPU et dépendances scientifiques | `requirements.txt`, `.venv-cuda/` | Environnement local présent ; installation historique non entièrement persistée |
| Acquisition | Lire des CSV locaux CIC-IDS2017/CIC-DDoS2019 | `data/raw/`, `src/ddos_tl/data/loading.py` | Données locales utilisées ; provenance amont non vérifiée |
| Source | Restreindre CIC-IDS2017 au CSV DDoS du vendredi et mapper `BENIGN`/`DDoS` | `configs/default.yaml`, `prepare_data.py` | Terminé |
| Nettoyage | Retirer identifiants, valeurs invalides, doublons et classes trop rares | `loading.py`, `pipeline.py` | Terminé et rapporté |
| Features | Harmoniser sémantiquement les colonnes vers 76 variables | `features.py`, métadonnées préparées | Terminé |
| Labels | Encoder les classes cibles et conserver leur ordre dans les métadonnées | `preprocessing.py`, `target_meta.json` | Terminé |
| Splits | Construire train/validation/test, par défaut par stratification aléatoire | `splitting.py`, `pipeline.py` | Terminé ; limite scientifique documentée |
| Préprocessing | Médiane, moyenne, échelle calculées sur le train, puis clipping | `preprocessing.py` | Terminé |
| MLP | Séparer backbone et tête pour le gel/dégel | `models/mlp.py` | Terminé |
| Pré-entraînement | Apprendre un backbone binaire sur la source | `scripts/pretrain.py`, `models/pretrained_*` | Terminé |
| Grille historique | Comparer Scratch, Frozen, Partial, Fine-tuning, Progressive, fractions et seeds | `scripts/run_experiments.py`, `experiments.csv` | 60 runs terminés |
| Contrôle 2×2 | Isoler initialisation et preprocessing avec 4 cellules, 4 fractions et 3 seeds | `experiments/`, `phase_2x2_scaler_initialization/` | 48 runs terminés |
| Audit 108 runs | Vérifier intégrité, classes, métriques et effets | `scripts/final_audit.py`, `final_audit_report.md` | Terminé |
| Validation statistique | Corriger les IC avec `t(2)=4.302653` et calculer les contrastes appariés | `scripts/statistical_validation.py` | Terminé dans un dossier complémentaire |
| Sélection finale | Sélectionner par Macro F1 validation, jamais par test | `scripts/export_final_model.py`, `final_training_plan.md` | Terminé |
| Entraînement final | Scratch 100 %, seed 0, early stopping validation | `models/best_model.pt` | Terminé |
| Évaluation finale | Une évaluation test post-sélection, métriques et matrice 18×18 | `scripts/evaluate_final_model.py`, `final_model/` | Terminé |
| Inference | Charger checkpoint et metadata, transformer et prédire | `inference.py`, `app/streamlit_app.py` | Terminé |
| Dashboard | Ajouter KPI, graphiques, filtres et exports | `app/streamlit_app.py`, `app_helpers.py` | Implémenté ; interaction navigateur complète à vérifier |

## Commandes historiques principales

Les commandes adaptées sont regroupées dans [COMMANDS_GUIDE.md](COMMANDS_GUIDE.md).
Elles ne sont pas exécutées par la génération de cette documentation.

