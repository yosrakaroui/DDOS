# Classification multiclasses des attaques DDoS par Transfer Learning

Mini-projet *Fundamentals of Deep Learning* (Chapter 5) : un MLP est **pré-entraîné** sur CIC-IDS2017, puis adapté à
**CIC-DDoS2019** (nouvelle tête, Freezing, Fine-tuning) et comparé à un entraînement from scratch, y compris quand les
données cibles sont rares. Une application Streamlit expose le meilleur modèle.

## Arborescence

```
ddos-transfer-learning/
├── README.md
├── requirements.txt · pyproject.toml · .gitignore
├── configs/
│   ├── default.yaml            # configuration principale (chemins, données, modèle, entraînement, expériences)
│   └── synthetic.yaml          # test rapide sur données synthétiques
├── data/
│   ├── raw/cic_ids2017/        # <- CSV source (pré-entraînement)
│   ├── raw/cic_ddos2019/       # <- CSV cible
│   └── processed/              # splits .npz + méta JSON (générés)
├── src/ddos_tl/
│   ├── config.py
│   ├── inference.py            # prédiction (entraînement + app)
│   ├── data/                   # loading, features, preprocessing, splitting, pipeline, synthetic
│   ├── models/mlp.py           # TabularMLP : backbone / tête, Freezing
│   ├── training/               # loops.py (boucle), strategies.py (scratch/frozen/partial/finetune/progressive)
│   ├── evaluation/             # metrics.py, plots.py
│   ├── tuning/search.py        # Optuna
│   └── utils/                  # seed, io
├── scripts/                    # prepare_data · pretrain · run_transfer · run_experiments · tune · plot_results · make_synthetic_data
├── notebooks/                  # 01 exploration · 02 préprocessing/pré-entraînement · 03 résultats
├── app/streamlit_app.py
├── tests/
├── models/                     # poids générés
├── reports/{results,figures}/  # résultats générés
└── docs/                       # architecture · protocole expérimental · plan du rapport
```

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate      Linux/macOS : source .venv/bin/activate
# 1) installer PyTorch selon ta machine (CPU/GPU) : https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## Test rapide (sans télécharger les datasets)

```bash
python scripts/make_synthetic_data.py --config configs/synthetic.yaml
python scripts/prepare_data.py        --config configs/synthetic.yaml
python scripts/pretrain.py            --config configs/synthetic.yaml
python scripts/run_experiments.py     --config configs/synthetic.yaml
python scripts/plot_results.py        --config configs/synthetic.yaml
pytest
```

## Utilisation sur les vrais datasets

1. Télécharger CIC-IDS2017 et CIC-DDoS2019 depuis le site du Canadian Institute for Cybersecurity (UNB)
   et placer les CSV dans `data/raw/cic_ids2017/` et `data/raw/cic_ddos2019/`.
   Dans cet environnement, les fichiers présents sont des miroirs locaux
   (CIC-IDS2017 depuis Hugging Face et CIC-DDoS2019 depuis un miroir local) :
   leur provenance officielle n'est pas considérée comme vérifiée.
2. **Ouvrir `notebooks/01_exploration.ipynb`** et vérifier : noms de colonnes réels, colonnes d'identification,
   répartition des classes par fichier. Adapter `configs/default.yaml` (`data.drop_columns`, `data.split.mode`...).
   Pour des fichiers très lourds, utiliser `data.sample_frac_per_file` (ex. `0.05`).
3. Lancer le pipeline :

```bash
python scripts/prepare_data.py                          # préparation contrôlée + rapports
# Génère aussi reports/results/data_report.json et data_report.md (préparation uniquement)
python scripts/pretrain.py                              # Phase 4  : pré-entraînement sur la source
python scripts/run_transfer.py --strategy scratch       # Phase 3  : baseline
python scripts/run_transfer.py --strategy finetune      # Phases 5-7
python scripts/run_experiments.py                       # Phases 8-10 : grille stratégies x fractions x seeds (reprise auto)
python scripts/tune.py --strategy finetune --trials 20  # Phase 9 (optionnelle)
python scripts/plot_results.py                          # figures + reports/results/summary.md
python scripts/run_transfer.py --strategy finetune --export-best
streamlit run app/streamlit_app.py                      # Phase 11
```

## Règles méthodologiques à respecter

- Le **test** n'est jamais utilisé pour choisir un modèle ou des hyperparamètres.
- Toujours rapporter **moyenne ± écart-type** sur plusieurs seeds, avec le **Macro F1** comme métrique principale.
- Ne jamais versionner les datasets (`.gitignore` s'en charge).
- Les résultats sur CIC-DDoS2019 ne garantissent pas la détection d'attaques réelles ou inconnues.

Voir `docs/` pour l'architecture détaillée, le protocole expérimental et le plan du rapport.
