"""Données synthétiques au format CIC (pour tester tout le pipeline sans télécharger les vrais datasets)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    " Flow Duration", " Total Fwd Packets", " Total Backward Packets", " Fwd Packet Length Mean",
    " Bwd Packet Length Mean", "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean", " Fwd IAT Mean",
    " Packet Length Mean", " Packet Length Std", " Average Packet Size", " Init_Win_bytes_forward",
    "Active Mean", " Idle Mean", " SYN Flag Count", " ACK Flag Count", " URG Flag Count",
    " Down/Up Ratio", " min_seg_size_forward",
]


def _make(rng, class_protos: dict, counts: dict, extra: str) -> pd.DataFrame:
    frames = []
    for label, n in counts.items():
        X = rng.lognormal(mean=class_protos[label], sigma=0.8, size=(n, len(FEATURES)))
        df = pd.DataFrame(X, columns=FEATURES)
        df[" Label"] = label
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    m = len(df)
    # colonnes d'identification et colonnes spécifiques à un dataset
    df["Flow ID"] = [f"10.0.{i % 255}.{i % 97}-8.8.8.8-{i}" for i in range(m)]
    df[" Source IP"] = "10.0.0.1"
    df[" Destination IP"] = "8.8.8.8"
    df[" Source Port"] = rng.integers(1024, 65535, m)
    df[" Destination Port"] = rng.integers(1, 1024, m)
    df[" Timestamp"] = "2019-03-11 10:00:00"
    df[extra] = rng.normal(size=m)
    # défauts typiques : infinis, valeurs manquantes, doublons
    inf_idx = rng.choice(m, size=max(1, m // 200), replace=False)
    df.loc[inf_idx, "Flow Bytes/s"] = np.inf
    nan_idx = rng.choice(m, size=max(1, m // 300), replace=False)
    df.loc[nan_idx, " Flow Packets/s"] = np.nan
    dup = df.sample(frac=0.02, random_state=1)
    return pd.concat([df, dup], ignore_index=True).sample(frac=1.0, random_state=2).reset_index(drop=True)


def generate(source_dir, target_dir, n_benign: int = 1200, n_attack: int = 400, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = len(FEATURES)
    p = {k: rng.normal(0.0, 1.5, n) for k in ("benign", "dos", "scan", "new")}
    src_dir, tgt_dir = Path(source_dir), Path(target_dir)
    src_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    # source : BENIGN, DoS, PortScan
    src_protos = {"BENIGN": p["benign"], "DoS": p["dos"], "PortScan": p["scan"]}
    src_counts = {"BENIGN": n_benign // 2, "DoS": n_attack // 2, "PortScan": n_attack // 2}
    for day in ("monday", "tuesday"):
        _make(rng, src_protos, src_counts, " Fwd Header Length.1").to_csv(src_dir / f"{day}.csv", index=False)

    # cible : attaques "proches" des attaques source + une classe nouvelle
    tgt_protos = {
        "BENIGN": p["benign"] + rng.normal(0, 0.1, n),
        "DrDoS_UDP": p["dos"] + rng.normal(0, 0.3, n),
        "Syn": p["scan"] + rng.normal(0, 0.3, n),
        "UDP-lag": p["new"],
    }
    tgt_counts = {"BENIGN": n_benign // 2, "DrDoS_UDP": n_attack // 2, "Syn": n_attack // 2, "UDP-lag": n_attack // 4}
    for day in ("train_day", "test_day"):
        df = _make(rng, tgt_protos, tgt_counts, "SimillarHTTP")
        df["Unnamed: 0"] = np.arange(len(df))
        df.to_csv(tgt_dir / f"{day}.csv", index=False)
