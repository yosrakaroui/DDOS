"""Phases 8 et 10 : grille stratégies x fractions de données x seeds (reprise automatique).

Sortie : <results>/experiments.csv  (une ligne par run) + <results>/runs/*.json (détails par classe)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config, resolve
from ddos_tl.data.pipeline import load_splits
from ddos_tl.training.strategies import run_strategy
from ddos_tl.utils.io import save_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = load_splits(resolve(cfg["paths"]["processed"]), "target")
    pretrained = resolve(cfg["paths"]["models"]) / "pretrained_backbone.pt"
    results_dir = resolve(cfg["paths"]["results"])
    csv_path = results_dir / "experiments.csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    done = set()
    rows = []
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        rows = old.to_dict("records")
        done = {(r["strategy"], float(r["fraction"]), int(r["seed"])) for r in rows}

    ex = cfg["experiments"]
    grid = [(s, f, sd) for s in ex["strategies"] for f in ex["fractions"] for sd in ex["seeds"]]
    for i, (strategy, frac, seed) in enumerate(grid, 1):
        if (strategy, float(frac), int(seed)) in done:
            continue
        if strategy != "scratch" and not pretrained.exists():
            raise SystemExit("Backbone pré-entraîné introuvable : lance d'abord scripts/pretrain.py")
        res = run_strategy(strategy, data, cfg, seed=seed, frac=frac,
                           pretrained_path=pretrained if strategy != "scratch" else None)
        tm = res["test_metrics"]
        rows.append({
            "strategy": strategy, "fraction": frac, "seed": seed, "n_train": res["n_train"],
            "val_macro_f1": res["val_macro_f1"], "test_accuracy": tm["accuracy"],
            "test_macro_precision": tm["macro_precision"], "test_macro_recall": tm["macro_recall"],
            "test_macro_f1": tm["macro_f1"], "test_weighted_f1": tm["weighted_f1"],
            "train_time_s": res["train_time_s"], "inference_us_per_sample": res["inference_us_per_sample"],
        })
        save_json({k: v for k, v in res.items() if k != "model"},
                  results_dir / "runs" / f"{strategy}_f{frac}_s{seed}.json")
        pd.DataFrame(rows).to_csv(csv_path, index=False)  # sauvegarde après chaque run
        print(f"[{i}/{len(grid)}] {strategy:12s} frac={frac:<5} seed={seed} -> test Macro F1 = {tm['macro_f1']:.4f}")
    print("Résultats :", csv_path)
