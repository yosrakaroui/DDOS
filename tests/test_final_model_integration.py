import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ddos_tl.config import resolve
from ddos_tl.data.pipeline import load_splits
from ddos_tl.data.preprocessing import Preprocessor
from ddos_tl.inference import predict_proba
from ddos_tl.models.mlp import TabularMLP


def test_exported_model_and_metadata_are_consistent():
    model_path = resolve("models/best_model.pt")
    meta_path = resolve("models/best_meta.json")
    assert model_path.exists()
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(meta["feature_columns"]) == 76
    assert len(meta["class_names"]) == 18
    assert meta["checkpoint_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = TabularMLP(
        meta["arch"]["in_dim"], len(meta["class_names"]),
        meta["arch"]["hidden"], meta["arch"]["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    assert model.head.out_features == 18


def test_exported_model_predicts_valid_probabilities_on_target_examples():
    meta = json.loads(resolve("models/best_meta.json").read_text(encoding="utf-8"))
    data = load_splits(resolve("data/processed"), "target")
    model = TabularMLP(
        meta["arch"]["in_dim"], len(meta["class_names"]),
        meta["arch"]["hidden"], meta["arch"]["dropout"],
    )
    model.load_state_dict(torch.load(resolve("models/best_model.pt"),
                                     map_location="cpu", weights_only=True))
    probabilities = predict_proba(model, data["X_test"][:8], torch.device("cpu"))
    assert probabilities.shape == (8, 18)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert set(probabilities.argmax(axis=1)).issubset(set(range(18)))


def test_exported_preprocessor_handles_invalid_input_without_nonfinite_output():
    meta = json.loads(resolve("models/best_meta.json").read_text(encoding="utf-8"))
    prep = Preprocessor.from_dict(meta["preprocessor"])
    import pandas as pd
    frame = pd.DataFrame({
        column: [np.nan, np.inf, "invalid"]
        for column in meta["feature_columns"]
    })
    transformed = prep.transform(frame)
    assert transformed.shape == (3, 76)
    assert np.isfinite(transformed).all()
    assert np.abs(transformed).max() <= 10
