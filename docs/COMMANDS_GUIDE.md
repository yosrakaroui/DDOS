# Guide des commandes

Toutes les commandes sont à exécuter depuis la racine du repository sous
PowerShell. Elles sont documentées ici mais ne sont pas lancées par cette
documentation.

## Environnement

```powershell
.\.venv-cuda\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

La première commande active l'environnement GPU local. La seconde installe
les dépendances déclarées ; PyTorch peut nécessiter une installation adaptée
à la version CUDA de la machine.

## Contrôle CUDA

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Données synthétiques

```powershell
python scripts\make_synthetic_data.py --config configs\synthetic.yaml
python scripts\prepare_data.py --config configs\synthetic.yaml
```

Ces commandes servent au test rapide et ne produisent pas de résultats
scientifiques réels.

## Préparation réelle

```powershell
python scripts\prepare_data.py --config configs\default.yaml
```

La commande lit les CSV locaux, applique le nettoyage, l'alignement, les
splits et le preprocessing. Elle produit les artefacts de `data/processed/`
et les rapports de données.

## Pré-entraînement et expériences

```powershell
python scripts\pretrain.py --config configs\default.yaml
python scripts\run_transfer.py --strategy scratch --frac 1.0 --seed 0 --config configs\default.yaml
python scripts\run_transfer.py --strategy finetune --frac 1.0 --seed 0 --config configs\default.yaml
python scripts\run_experiments.py --config configs\default.yaml
```

Ces commandes entraînent réellement. Elles ne doivent pas être relancées
sans décision scientifique et sauvegarde des sorties.

## Expérience 2×2

```powershell
python scripts\prepare_2x2_experiment.py --config configs\default.yaml
python scripts\run_2x2_experiment.py --dry-run --config configs\default.yaml
python scripts\run_2x2_experiment.py --execute --config configs\default.yaml
```

`--dry-run` valide les 48 configurations sans entraîner. `--execute` lance
les entraînements et écrit uniquement dans le répertoire contrôlé prévu.

## Audits

```powershell
python scripts\analyze_results.py
python scripts\plot_results.py --config configs\default.yaml
python scripts\final_audit.py
python scripts\statistical_validation.py
```

Ces scripts agrègent et vérifient les résultats existants. Ils ne doivent pas
être confondus avec un entraînement.

## Modèle final

```powershell
python scripts\export_final_model.py --config configs\default.yaml --execute
python scripts\evaluate_final_model.py --config configs\default.yaml
```

La première commande entraîne et exporte ; la seconde effectue l'évaluation
test post-sélection. Dans l'état actuel, les artefacts finaux sont déjà
présents et ces commandes ne sont pas nécessaires pour la documentation.

## Tests et application

```powershell
python -m pytest -q
python -m py_compile app\streamlit_app.py src\ddos_tl\app_helpers.py
python -m streamlit run app\streamlit_app.py
```

Le résultat vérifié actuel est 47 tests passés, compilation réussie et
endpoint Streamlit `/_stcore/health` répondant `200 ok`.

