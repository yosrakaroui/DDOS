"""Phases 3, 5, 6, 7 : entraîne UNE stratégie (baseline, frozen, partial, finetune, progressive).

Exemples :
  python scripts/run_transfer.py --strategy scratch                 # baseline DNN
  python scripts/run_transfer.py --strategy finetune --seed 0
  python scripts/run_transfer.py --strategy finetune --export-best  # modèle servi par l'app Web
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config, resolve
from ddos_tl.data.pipeline import load_splits
from ddos_tl.training.strategies import STRATEGIES, run_strategy
from ddos_tl.utils.io import save_json

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction du train cible utilisée")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export-best", action="store_true", help="exporte models/best_model.pt + best_meta.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = load_splits(resolve(cfg["paths"]["processed"]), "target")
    models_dir = resolve(cfg["paths"]["models"])
    pretrained = models_dir / "pretrained_backbone.pt"
    if args.strategy != "scratch" and not pretrained.exists():
        raise SystemExit("Backbone pré-entraîné introuvable : lance d'abord scripts/pretrain.py")

    res = run_strategy(args.strategy, data, cfg, seed=args.seed, frac=args.frac,
                       pretrained_path=pretrained if args.strategy != "scratch" else None, verbose=True)
    tm = res["test_metrics"]
    print(f"\n{args.strategy} | frac={args.frac} seed={args.seed} | n_train={res['n_train']} | "
          f"val F1={res['val_macro_f1']:.4f} | test Macro F1={tm['macro_f1']:.4f} acc={tm['accuracy']:.4f} | "
          f"train {res['train_time_s']:.1f}s")

    tag = f"{args.strategy}_f{args.frac}_s{args.seed}"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(res["model"].state_dict(), models_dir / f"{tag}.pt")
    save_json({k: v for k, v in res.items() if k != "model"}, resolve(cfg["paths"]["results"]) / "runs" / f"{tag}.json")

    if args.export_best:
        torch.save(res["model"].state_dict(), models_dir / "best_model.pt")
        save_json({
            "strategy": args.strategy, "arch": res["model"].arch, "feature_columns": data["feature_columns"],
            "class_names": data["class_names"], "preprocessor": data["preprocessor"],
            "feature_alignment": data.get("feature_alignment"),
            "scaler_type": res["scaler_type"],
            "scaler_fit_dataset": res["scaler_fit_dataset"],
            "scaler_fit_split": res["scaler_fit_split"],
            "scaler_mean": res["scaler_mean"],
            "scaler_scale": res["scaler_scale"],
            "clipping": res["clipping"],
            "preprocessing_strategy": res["preprocessing_strategy"],
            "test_metrics": {k: tm[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
            "per_class": tm["per_class"],
        }, models_dir / "best_meta.json")
        print("Modèle exporté pour l'application :", models_dir / "best_model.pt")
