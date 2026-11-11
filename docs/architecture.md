# Architecture du projet

## Flux de données et de modèles

```
CIC-IDS2017 (source)           CIC-DDoS2019 (cible)
        │                              │
        └──── prepare_data.py ─────────┘   nettoyage · retrait des colonnes d'identification ·
                     │                      alignement sémantique canonique · split · normalisation
                     ▼                      (statistiques calculées sur le train uniquement)
   data/processed/{source,target}_splits.npz + *_meta.json
                     │
          pretrain.py (source)  ──►  models/pretrained_backbone.pt + pretrained_meta.json
                     │
   run_transfer.py / run_experiments.py (cible)
        scratch · frozen · partial · finetune · progressive
                     │
                     ▼
   reports/results/experiments.csv + runs/*.json ──► plot_results.py ──► figures + summary.md
                     │
        run_transfer.py --export-best ──► models/best_model.pt + best_meta.json ──► app/streamlit_app.py
```

## Correspondance phases du cahier des charges ↔ fichiers

| Phase | Contenu | Fichiers |
|---|---|---|
| 1 | Exploration | `notebooks/01_exploration.ipynb` |
| 2 | Préprocessing | `scripts/prepare_data.py`, `src/ddos_tl/data/` |
| 3 | Baseline DNN (from scratch) | `scripts/run_transfer.py --strategy scratch`, `src/ddos_tl/models/mlp.py` |
| 4 | Pré-entraînement sur la source | `scripts/pretrain.py` |
| 5 | Transfer Learning (nouvelle tête) | `TabularMLP.load_backbone / replace_head`, `training/strategies.py` |
| 6 | Freezing | `TabularMLP.freeze_blocks`, stratégies `frozen`, `partial`, `progressive` |
| 7 | Fine-tuning | stratégie `finetune` (tête seule, puis tout débloqué à lr réduit) |
| 8 | Étude low-data | `scripts/run_experiments.py` (`experiments.fractions`) |
| 9 | Optuna (optionnel) | `scripts/tune.py`, `src/ddos_tl/tuning/search.py` |
| 10 | Évaluation et comparaison | `plot_results.py`, `notebooks/03_results_comparison.ipynb`, `evaluation/` |
| 11 | Application Web | `app/streamlit_app.py` |
| 12 | Tests et documentation | `tests/`, `README.md`, `docs/` |

## Stratégies comparées

| Stratégie | Backbone | Entraînement |
|---|---|---|
| `scratch` | aléatoire | tout, lr normal (baseline) |
| `frozen` | pré-entraîné, **gelé** | tête seule |
| `partial` | pré-entraîné, k premiers blocs gelés | blocs restants à lr réduit + tête |
| `finetune` | pré-entraîné | étape 1 : tête seule ; étape 2 : tout débloqué à lr réduit |
| `progressive` | pré-entraîné | tête, puis dégel bloc par bloc (du haut vers le bas) |

## Décisions de conception

- **Pas de pickle** : le préprocesseur est sérialisé en JSON, les poids en `state_dict` chargés avec `weights_only=True`.
- **Backbone / tête séparés** (`blocks` / `head`) : c'est ce qui rend le Freezing et le remplacement de la couche de sortie propres.
- **Macro F1 partout** : même définition pour l'early stopping, l'évaluation et Optuna (classes présentes dans `y_true`).
- **Optuna ne règle pas l'architecture** : elle est imposée par le backbone pré-entraîné.
- **Reprise automatique** de `run_experiments.py` : un run déjà présent dans `experiments.csv` est sauté.
- **Labels source conservés** : `BENIGN` et `DDoS` restent dans les métadonnées,
  tandis que les indices numériques sont utilisés par PyTorch.
- **Application cohérente** : Streamlit applique le mapping canonique avant le preprocessing.
- **Rapports** : `prepare_data.py` produit `data_report.json` et `data_report.md` sans entraîner de modèle.
