"""Optimisation des hyperparamètres avec Optuna (sur la validation uniquement, jamais sur le test).

L'architecture (hidden) est figée par le backbone pré-entraîné : on ne règle donc que
les hyperparamètres d'entraînement (lr, weight decay, batch size, facteur de lr du backbone, dropout).
"""
from __future__ import annotations

import copy

import optuna

from ..training.strategies import run_strategy


def make_objective(strategy: str, data: dict, cfg: dict, pretrained_path, seed: int = 0):
    def objective(trial: optuna.Trial) -> float:
        c = copy.deepcopy(cfg)
        c["training"]["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        c["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        c["training"]["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
        c["transfer"]["finetune_lr_factor"] = trial.suggest_float("finetune_lr_factor", 0.02, 0.5, log=True)
        # Le dropout ne change pas la forme des poids : compatible avec le backbone pré-entraîné.
        c["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        res = run_strategy(strategy, data, c, seed=seed, pretrained_path=pretrained_path)
        return res["val_macro_f1"]
    return objective


def run_search(strategy: str, data: dict, cfg: dict, pretrained_path, n_trials: int, seed: int = 0):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(make_objective(strategy, data, cfg, pretrained_path, seed), n_trials=n_trials)
    return study
