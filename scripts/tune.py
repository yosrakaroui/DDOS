"""Phase 9 (optionnelle) : Optuna, optimisation sur la VALIDATION uniquement.

Usage : python scripts/tune.py --strategy finetune --trials 20
Sorties : <results>/best_params_<strategy>.json, <results>/optuna_<strategy>.csv
Ensuite, reporte les meilleurs paramètres dans configs/default.yaml et relance run_experiments.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config, resolve
from ddos_tl.data.pipeline import load_splits
from ddos_tl.training.strategies import STRATEGIES
from ddos_tl.tuning.search import run_search
from ddos_tl.utils.io import save_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--strategy", default="finetune", choices=STRATEGIES)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = load_splits(resolve(cfg["paths"]["processed"]), "target")
    pretrained = resolve(cfg["paths"]["models"]) / "pretrained_backbone.pt"
    study = run_search(args.strategy, data, cfg, pretrained if args.strategy != "scratch" else None,
                       args.trials, cfg["seed"])
    out = resolve(cfg["paths"]["results"])
    save_json({"best_val_macro_f1": study.best_value, "best_params": study.best_params},
              out / f"best_params_{args.strategy}.json")
    study.trials_dataframe().to_csv(out / f"optuna_{args.strategy}.csv", index=False)
    print("Meilleurs paramètres :", study.best_params, "| val Macro F1 =", round(study.best_value, 4))
