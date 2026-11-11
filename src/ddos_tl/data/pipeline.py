"""Du DataFrame nettoyé aux tableaux numpy prêts pour l'entraînement."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import load_json, save_json
from .loading import LABEL_TEXT_COLUMN
from .preprocessing import LabelMap, Preprocessor
from .splitting import by_file_split, cap_per_class, drop_rare_classes, stratified_split

SPLITS = ("train", "val", "test")
FORBIDDEN_FEATURE_TOKENS = {
    "label", "class", "unnamed: 0", "flow id", "source ip",
    "destination ip", "timestamp", "source port", "destination port",
}


def validate_feature_alignment(feature_columns: list[str], alignment: dict | None = None) -> None:
    """Validate the final feature contract used by source and target."""
    if not feature_columns:
        raise ValueError("Aucune feature alignée n'est disponible.")
    lowered = {str(column).strip().lower() for column in feature_columns}
    forbidden = sorted(lowered & FORBIDDEN_FEATURE_TOKENS)
    if forbidden:
        raise ValueError(f"Feature(s) interdite(s) dans le modèle: {forbidden}")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Les features finales contiennent des doublons.")
    if alignment is not None:
        aligned = alignment.get("common", [])
        if list(feature_columns) != list(aligned):
            raise ValueError("L'ordre des features ne correspond pas à l'alignement.")


def validate_split_disjointness(parts: tuple[pd.DataFrame, ...]) -> dict:
    """Confirm that train, validation and test have no shared source rows."""
    names = ("train", "val", "test")
    indices = {name: set(part.index.tolist()) for name, part in zip(names, parts)}
    overlaps = {
        f"{left}_{right}": sorted(indices[left] & indices[right])
        for i, left in enumerate(names) for right in names[i + 1:]
    }
    if any(overlaps.values()):
        raise ValueError(f"Chevauchement entre splits détecté: {overlaps}")
    return {"overlap_counts": {key: len(value) for key, value in overlaps.items()}}


def build_splits(df: pd.DataFrame, feature_cols: list[str], label_col: str, data_cfg: dict,
                 seed: int, force_random: bool = False,
                 feature_alignment: dict | None = None,
                 label_mapping: dict[str, int] | None = None,
                 preprocessor: Preprocessor | None = None,
                 scaler_type: str = "standard",
                 scaler_fit_dataset: str = "current",
                 scaler_fit_split: str = "train",
                 preprocessing_strategy: str = "fit_on_current_train") -> dict:
    class_counts_before_filter = {
        str(k): int(v) for k, v in df[label_col].value_counts().items()
    }
    df = drop_rare_classes(df, label_col, data_cfg["min_rows_per_class"])
    class_counts_after_filter = {
        str(k): int(v) for k, v in df[label_col].value_counts().items()
    }
    df = cap_per_class(df, label_col, data_cfg["max_rows_per_class"], seed)
    class_counts_after_cap = {
        str(k): int(v) for k, v in df[label_col].value_counts().items()
    }

    sp = data_cfg["split"]
    if sp["mode"] == "by_file" and not force_random:
        parts = by_file_split(df, label_col, sp.get("train_files", []), sp.get("test_files", []),
                              sp["val_size"], seed)
    else:
        parts = stratified_split(df, label_col, sp["val_size"], sp["test_size"], seed)
    train_df, val_df, test_df = parts

    validate_feature_alignment(feature_cols, feature_alignment)
    split_checks = validate_split_disjointness(parts)
    if preprocessor is None:
        prep = Preprocessor(feature_cols, clip=data_cfg.get("clip", 10.0)).fit(train_df)
    else:
        if preprocessor.feature_columns != list(feature_cols):
            raise ValueError("Le preprocessor fourni n'a pas le même ordre de features.")
        prep = preprocessor
    if label_mapping is not None:
        classes = [name for name, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
        labels = LabelMap(classes)
        label_values = df[LABEL_TEXT_COLUMN].astype(str)
    else:
        labels = LabelMap.fit(df[label_col])
        label_values = df[label_col].astype(str)

    out: dict = {"preprocessor": prep, "label_map": labels, "feature_columns": list(feature_cols),
                 "class_names": labels.classes, "counts": {},
                 "split_metadata": {
                     "mode": "random" if force_random else sp["mode"],
                     "seed": seed,
                     "val_size": sp["val_size"],
                     "test_size": sp["test_size"],
                     "preprocessor_fit_split": "train",
                     "scaler_type": scaler_type,
                     "scaler_fit_dataset": scaler_fit_dataset,
                     "scaler_fit_split": scaler_fit_split,
                     "preprocessing_strategy": preprocessing_strategy,
                     "class_counts_before_filter": class_counts_before_filter,
                     "class_counts_after_rare_filter": class_counts_after_filter,
                     "class_counts_after_cap": class_counts_after_cap,
                     "rare_classes_removed": sorted(
                         set(class_counts_before_filter) - set(class_counts_after_filter)
                     ),
                     **split_checks,
                 }}
    for name, part in zip(SPLITS, parts):
        out[f"X_{name}"] = prep.transform(part)
        out[f"y_{name}"] = labels.encode(
            part[LABEL_TEXT_COLUMN] if label_mapping is not None else part[label_col]
        )
        out["counts"][name] = {
            str(k): int(v) for k, v in label_values.loc[part.index].value_counts().items()
        }
    if label_mapping is not None:
        out["label_mapping"] = label_mapping
        out["label_text_column"] = LABEL_TEXT_COLUMN
    out["preprocessing_metadata"] = {
        "scaler_type": scaler_type,
        "scaler_fit_dataset": scaler_fit_dataset,
        "scaler_fit_split": scaler_fit_split,
        "clipping": prep.clip,
        "feature_columns": list(feature_cols),
        "preprocessing_strategy": preprocessing_strategy,
        "scaler_mean": prep.mean_.tolist(),
        "scaler_scale": prep.scale_.tolist(),
    }
    return out


def save_splits(directory: str | Path, name: str, splits: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value for key, value in splits.items()
        if (key.startswith("X_") or key.startswith("y_"))
        and isinstance(value, np.ndarray)
    }
    np.savez_compressed(directory / f"{name}_splits.npz", **arrays)
    save_json({
        "feature_columns": splits["feature_columns"],
        "class_names": splits["class_names"],
        "preprocessor": splits["preprocessor"].to_dict(),
        "preprocessing_metadata": splits.get("preprocessing_metadata", {}),
        "counts": splits["counts"],
        "split_metadata": splits.get("split_metadata", {}),
        **({"label_mapping": splits["label_mapping"]} if "label_mapping" in splits else {}),
        **({"feature_alignment": splits["feature_alignment"]}
           if "feature_alignment" in splits else {}),
        **({"transfer_preprocessor": splits["transfer_preprocessor"]}
           if "transfer_preprocessor" in splits else {}),
    }, directory / f"{name}_meta.json")


def load_splits(directory: str | Path, name: str) -> dict:
    directory = Path(directory)
    npz_path, meta_path = directory / f"{name}_splits.npz", directory / f"{name}_meta.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} introuvable : lance d'abord scripts/prepare_data.py")
    arrays = np.load(npz_path)
    meta = load_json(meta_path)
    out = {k: arrays[k] for k in arrays.files}
    out.update(feature_columns=meta["feature_columns"], class_names=meta["class_names"],
               preprocessor=meta["preprocessor"], counts=meta["counts"],
               split_metadata=meta.get("split_metadata", {}),
               preprocessing_metadata=meta.get("preprocessing_metadata", {}))
    if "label_mapping" in meta:
        out["label_mapping"] = meta["label_mapping"]
    if "feature_alignment" in meta:
        out["feature_alignment"] = meta["feature_alignment"]
    if "transfer_preprocessor" in meta:
        out["transfer_preprocessor"] = meta["transfer_preprocessor"]
    return out
