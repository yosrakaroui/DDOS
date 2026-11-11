"""Sélection, documentation et alignement sémantique des features CIC."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .loading import AUX_COLUMN


# Ces correspondances sont fondées sur les dictionnaires de features
# CICFlowMeter : seule la formulation change, pas la quantité mesurée.
SEMANTIC_FEATURE_MAPPING = (
    {
        "source": "Total Length of Fwd Packets",
        "target": "Fwd Packets Length Total",
        "canonical": "fwd_packets_length_total",
        "reason": "total length of forward packets",
    },
    {
        "source": "Total Length of Bwd Packets",
        "target": "Bwd Packets Length Total",
        "canonical": "bwd_packets_length_total",
        "reason": "total length of backward packets",
    },
    {
        "source": "Average Packet Size",
        "target": "Avg Packet Size",
        "canonical": "avg_packet_size",
        "reason": "average packet size",
    },
    {
        "source": "Max Packet Length",
        "target": "Packet Length Max",
        "canonical": "packet_length_max",
        "reason": "maximum packet length",
    },
    {
        "source": "Min Packet Length",
        "target": "Packet Length Min",
        "canonical": "packet_length_min",
        "reason": "minimum packet length",
    },
    {
        "source": "Init_Win_bytes_forward",
        "target": "Init Fwd Win Bytes",
        "canonical": "init_fwd_win_bytes",
        "reason": "initial forward TCP window bytes",
    },
    {
        "source": "Init_Win_bytes_backward",
        "target": "Init Bwd Win Bytes",
        "canonical": "init_bwd_win_bytes",
        "reason": "initial backward TCP window bytes",
    },
    {
        "source": "act_data_pkt_fwd",
        "target": "Fwd Act Data Packets",
        "canonical": "fwd_act_data_packets",
        "reason": "forward packets carrying active data",
    },
    {
        "source": "min_seg_size_forward",
        "target": "Fwd Seg Size Min",
        "canonical": "fwd_seg_size_min",
        "reason": "minimum forward segment size",
    },
)

EXCLUDED_FEATURE_REASONS = {
    "label": "target label",
    "class": "binary target label",
    "unnamed: 0": "index column",
    "flow id": "flow identifier",
    "source ip": "endpoint identifier",
    "destination ip": "endpoint identifier",
    "timestamp": "capture timestamp",
    "simillarhttp": "application identifier / potential leakage",
    "source port": "endpoint identifier not shared by both datasets",
    "destination port": "endpoint identifier not shared by both datasets",
    "protocol": "no source counterpart in the selected CIC-IDS2017 file",
}


def numeric_feature_columns(df: pd.DataFrame, label_column: str) -> list[str]:
    return [
        c for c in df.columns
        if c not in (label_column, AUX_COLUMN) and pd.api.types.is_numeric_dtype(df[c])
    ]


def validate_feature_mapping(mapping: Iterable[dict]) -> None:
    """Reject duplicate source, target or canonical names in the mapping."""
    entries = list(mapping)
    for key in ("source", "target", "canonical"):
        values = [entry[key] for entry in entries]
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"Doublon dans le mapping '{key}': {duplicates}")


def _excluded(column: str, side: str) -> dict:
    reason = EXCLUDED_FEATURE_REASONS.get(column.strip().lower())
    return {
        "side": side,
        "original_name": column,
        "canonical_name": None,
        "reason": reason or "no validated semantic counterpart",
    }


def align_features(source_cols: list[str], target_cols: list[str],
                   mapping: Iterable[dict] = SEMANTIC_FEATURE_MAPPING) -> dict:
    """Build a deterministic source/target feature alignment.

    Exact matches are retained only when both datasets expose the same name.
    Renamed matches must be present in ``mapping``; similar-looking names are
    never paired implicitly.
    """
    validate_feature_mapping(mapping)
    source = list(dict.fromkeys(source_cols))
    target = list(dict.fromkeys(target_cols))
    source_set, target_set = set(source), set(target)
    pairs = []
    used_source, used_target = set(), set()

    for column in sorted(source_set & target_set):
        if column.strip().lower() in EXCLUDED_FEATURE_REASONS:
            continue
        pairs.append({
            "source_original_name": column,
            "target_original_name": column,
            "canonical_name": column,
            "reason": "same feature name and definition",
        })
        used_source.add(column)
        used_target.add(column)

    for entry in mapping:
        src, tgt = entry["source"], entry["target"]
        if src in source_set and tgt in target_set:
            if src in used_source or tgt in used_target:
                raise ValueError(f"Mapping en conflit pour '{src}' / '{tgt}'")
            pairs.append({
                "source_original_name": src,
                "target_original_name": tgt,
                "canonical_name": entry["canonical"],
                "reason": entry["reason"],
            })
            used_source.add(src)
            used_target.add(tgt)

    canonical_names = [pair["canonical_name"] for pair in pairs]
    if len(canonical_names) != len(set(canonical_names)):
        raise ValueError("Collision entre noms canoniques dans l'alignement")
    pairs.sort(key=lambda pair: pair["canonical_name"])
    excluded = [
        _excluded(column, "source")
        for column in sorted(source_set - used_source)
    ] + [
        _excluded(column, "target")
        for column in sorted(target_set - used_target)
    ]
    return {
        "common": [pair["canonical_name"] for pair in pairs],
        "pairs": pairs,
        "excluded": excluded,
        "source_only": sorted(source_set - used_source),
        "target_only": sorted(target_set - used_target),
    }


def apply_alignment(df: pd.DataFrame, alignment: dict, side: str) -> pd.DataFrame:
    """Rename aligned original columns to the shared canonical feature names."""
    if side not in ("source", "target"):
        raise ValueError("side doit être 'source' ou 'target'")
    rename = {
        pair[f"{side}_original_name"]: pair["canonical_name"]
        for pair in alignment["pairs"]
    }
    return df.rename(columns=rename)
