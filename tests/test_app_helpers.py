from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from ddos_tl.app_helpers import (
    build_prediction_frame,
    evaluate_labeled_frame,
    load_uploaded_csv,
    transform_for_inference,
    validate_input_columns,
)
from ddos_tl.data.preprocessing import Preprocessor


class Upload:
    def __init__(self, value: bytes):
        self.value = value

    def getvalue(self):
        return self.value


def prep():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    return Preprocessor(["a", "b"], clip=10).fit(train)


def test_valid_csv_and_missing_columns():
    uploaded = Upload(b"a,b\n1,2\n3,4\n")
    frame = load_uploaded_csv(uploaded, 10)
    assert frame.shape == (2, 2)
    incomplete = load_uploaded_csv(Upload(b"a\n1\n"), 10)
    assert validate_input_columns(incomplete, prep()) == ["b"]


def test_empty_nan_and_inf_are_handled_without_nonfinite_output():
    with pytest.raises(ValueError):
        load_uploaded_csv(Upload(b""), 10)
    frame = pd.DataFrame({"a": [np.nan, np.inf], "b": [-np.inf, 2.0]})
    transformed, quality = transform_for_inference(frame, prep())
    assert quality["invalid_cells"] == 3
    assert np.isfinite(transformed).all()


def test_multiclass_probabilities_and_prediction_table():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
    probabilities = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
    result = build_prediction_frame(frame, probabilities, ["Benign", "Syn", "UDP"])
    assert list(result["Classe prédite"]) == ["Benign", "UDP"]
    assert list(result["Niveau"]) == ["Élevée", "Moyenne"]
    assert len(result.filter(like="P(").columns) == 3


def test_labels_are_evaluated_only_when_present_and_known():
    frame = pd.DataFrame({"Label": ["Benign", "Syn", "unknown"]})
    metrics, unknown = evaluate_labeled_frame(frame, np.array([0, 1, 1]), ["Benign", "Syn"], "Label")
    assert metrics["accuracy"] == 1.0
    assert unknown == 1
    absent, unknown = evaluate_labeled_frame(pd.DataFrame({"x": [1]}), np.array([0]), ["Benign"], "Label")
    assert absent is None and unknown == 0
