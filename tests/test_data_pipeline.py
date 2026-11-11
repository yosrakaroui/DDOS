import numpy as np
import pandas as pd
import pytest

from ddos_tl.data.features import (
    SEMANTIC_FEATURE_MAPPING,
    align_features,
    apply_alignment,
    numeric_feature_columns,
    validate_feature_mapping,
)
from ddos_tl.data.loading import (
    AUX_COLUMN,
    LABEL_TEXT_COLUMN,
    apply_label_mapping,
    clean_frame,
    read_csv_dir,
)
from ddos_tl.data.pipeline import (
    build_splits,
    load_splits,
    save_splits,
    validate_feature_alignment,
    validate_split_disjointness,
)
from ddos_tl.data.preprocessing import LabelMap, Preprocessor
from ddos_tl.data.splitting import by_file_split, subsample_train
from scripts.prepare_data import report_markdown


def _clean(directory, cfg):
    raw = read_csv_dir(directory)
    return clean_frame(raw, cfg["data"]["label_column"], cfg["data"]["drop_columns"])


def test_cleaning_removes_ids_infinities_and_duplicates(synthetic_dirs, cfg):
    _, tgt_dir = synthetic_dirs
    raw = read_csv_dir(tgt_dir)
    df = clean_frame(raw, "Label", cfg["data"]["drop_columns"])
    for col in ("Flow ID", "Source IP", "Destination IP", "Source Port", "Destination Port", "Timestamp",
                "SimillarHTTP", "Unnamed: 0"):
        assert col not in df.columns
    assert AUX_COLUMN in df.columns
    feats = numeric_feature_columns(df, "Label")
    assert len(feats) == 20
    assert not np.isinf(df[feats].to_numpy()).any()
    assert len(df) < len(raw)  # doublons retirés
    assert set(df["Label"]) == {"BENIGN", "DrDoS_UDP", "Syn", "UDP-lag"}


def test_alignment_keeps_only_common_features(synthetic_dirs, cfg):
    src_dir, tgt_dir = synthetic_dirs
    src, tgt = _clean(src_dir, cfg), _clean(tgt_dir, cfg)
    al = align_features(numeric_feature_columns(src, "Label"), numeric_feature_columns(tgt, "Label"))
    assert len(al["common"]) == 20
    assert "Fwd Header Length.1" in al["source_only"]


def test_semantic_alignment_uses_documented_pairs_and_order():
    source = ["Flow Duration", "Total Length of Fwd Packets", "Unknown Source"]
    target = ["Flow Duration", "Fwd Packets Length Total", "Unknown Target"]
    al = align_features(source, target)
    assert al["common"] == ["Flow Duration", "fwd_packets_length_total"]
    assert al["pairs"][1]["source_original_name"] == "Total Length of Fwd Packets"
    assert al["pairs"][1]["target_original_name"] == "Fwd Packets Length Total"
    assert al["pairs"][1]["canonical_name"] == "fwd_packets_length_total"
    assert all(item["reason"] == "no validated semantic counterpart"
               for item in al["excluded"])

    src_df = pd.DataFrame({"Total Length of Fwd Packets": [1.0], "Flow Duration": [2.0]})
    tgt_df = pd.DataFrame({"Fwd Packets Length Total": [1.0], "Flow Duration": [2.0]})
    assert list(apply_alignment(src_df, al, "source").columns) == [
        "fwd_packets_length_total", "Flow Duration"
    ]
    assert list(apply_alignment(tgt_df, al, "target").columns) == [
        "fwd_packets_length_total", "Flow Duration"
    ]


def test_alignment_reports_missing_semantic_counterpart():
    al = align_features(
        ["Total Length of Fwd Packets"],
        ["Fwd Packets Length Total"],
        mapping=[{
            "source": "Missing Source",
            "target": "Fwd Packets Length Total",
            "canonical": "missing",
            "reason": "unverified",
        }],
    )
    assert al["common"] == []
    assert al["source_only"] == ["Total Length of Fwd Packets"]
    assert al["target_only"] == ["Fwd Packets Length Total"]


def test_alignment_rejects_duplicate_mapping_entries():
    duplicate = list(SEMANTIC_FEATURE_MAPPING) + [{
        "source": "Other Source",
        "target": "Other Target",
        "canonical": "fwd_seg_size_min",
        "reason": "duplicate canonical",
    }]
    try:
        validate_feature_mapping(duplicate)
        assert False, "doit refuser les canoniques dupliqués"
    except ValueError as exc:
        assert "canonical" in str(exc)


def test_preprocessor_is_fitted_on_train_only_and_roundtrips():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(10, 2, 500), "b": rng.normal(-5, 1, 500)})
    test = pd.DataFrame({"a": rng.normal(100, 2, 50), "b": [np.inf] * 50})
    prep = Preprocessor(["a", "b"], clip=10).fit(train)
    assert abs(prep.mean_[0] - train["a"].mean()) < 1e-9          # statistiques du train seulement
    Z = prep.transform(test)
    assert np.isfinite(Z).all() and np.abs(Z).max() <= 10          # inf imputé, valeurs bornées
    again = Preprocessor.from_dict(prep.to_dict())
    np.testing.assert_allclose(again.transform(test), Z)


def test_label_map_roundtrip_and_unknown_label():
    lm = LabelMap.fit(["Syn", "BENIGN", "Syn"])
    assert lm.classes == ["BENIGN", "Syn"]
    assert lm.decode(lm.encode(["Syn", "BENIGN"])) == ["Syn", "BENIGN"]
    try:
        lm.encode(["???"])
        assert False, "doit lever ValueError"
    except ValueError:
        pass


def test_splits_are_stratified_and_saved(synthetic_dirs, cfg, tmp_path):
    _, tgt_dir = synthetic_dirs
    df = _clean(tgt_dir, cfg)
    cols = numeric_feature_columns(df, "Label")
    s = build_splits(df, cols, "Label", cfg["data"], seed=0)
    n_classes = len(s["class_names"])
    for part in ("train", "val", "test"):
        assert set(np.unique(s[f"y_{part}"])) == set(range(n_classes))   # toutes les classes partout
        assert s[f"X_{part}"].shape[1] == len(cols)
    assert len(s["y_train"]) > len(s["y_val"]) and len(s["y_train"]) > len(s["y_test"])

    s["feature_alignment"] = {
        "pairs": [{"source_original_name": "a", "target_original_name": "b",
                   "canonical_name": "a_canonical", "reason": "verified"}],
        "excluded": [],
    }
    save_splits(tmp_path, "target", s)
    back = load_splits(tmp_path, "target")
    np.testing.assert_array_equal(back["y_test"], s["y_test"])
    assert back["class_names"] == s["class_names"]
    assert back["feature_alignment"] == s["feature_alignment"]


def test_subsample_train_is_stratified():
    y = np.repeat([0, 1, 2], [1000, 200, 50])
    X = np.zeros((len(y), 3), dtype=np.float32)
    _, ys = subsample_train(X, y, 0.05, seed=0)
    assert set(np.unique(ys)) == {0, 1, 2}
    assert (ys == 2).sum() >= 5


def test_source_file_selection_and_binary_label_mapping(tmp_path):
    first = pd.DataFrame({"Label": ["BENIGN", "DDoS"], "Feature": [1, 2]})
    second = pd.DataFrame({"Label": ["Other"], "Feature": [3]})
    first.to_csv(tmp_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", index=False)
    second.to_csv(tmp_path / "Tuesday-WorkingHours.pcap_ISCX.csv", index=False)

    raw = read_csv_dir(
        tmp_path,
        selected_files=["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
    )
    mapped = apply_label_mapping(raw, "Label", {"BENIGN": 0, "DDoS": 1})
    assert len(mapped) == 2
    assert mapped["Label"].tolist() == [0, 1]


def test_binary_label_mapping_rejects_unknown_labels():
    df = pd.DataFrame({"Label": ["BENIGN", "Other"]})
    try:
        apply_label_mapping(df, "Label", {"BENIGN": 0, "DDoS": 1})
        assert False, "doit refuser les labels source inconnus"
    except ValueError as exc:
        assert "Other" in str(exc)


def test_source_labels_preserve_text_and_numeric_indices():
    df = pd.DataFrame({"Label": ["DDoS", "BENIGN"], "Feature": [1, 2]})
    mapped = apply_label_mapping(df, "Label", {"BENIGN": 0, "DDoS": 1})
    assert mapped["Label"].tolist() == [1, 0]
    assert mapped[LABEL_TEXT_COLUMN].tolist() == ["DDoS", "BENIGN"]

    clean = clean_frame(mapped, "Label", [])
    assert LABEL_TEXT_COLUMN in clean.columns


def test_source_split_uses_original_class_names(synthetic_dirs, cfg):
    src_dir, _ = synthetic_dirs
    raw = read_csv_dir(src_dir)
    source = clean_frame(raw, "Label", cfg["data"]["drop_columns"])
    source = apply_label_mapping(
        source, "Label", {"BENIGN": 0, "DoS": 1, "PortScan": 2}
    )
    cols = numeric_feature_columns(source, "Label")
    split = build_splits(
        source, cols, "Label", cfg["data"], 0,
        label_mapping={"BENIGN": 0, "DoS": 1, "PortScan": 2},
    )
    assert split["class_names"] == ["BENIGN", "DoS", "PortScan"]
    assert split["label_mapping"] == {"BENIGN": 0, "DoS": 1, "PortScan": 2}
    assert set(split["counts"]["train"]) == {"BENIGN", "DoS", "PortScan"}


def test_scaler_metadata_and_transfer_variant_are_explicit(synthetic_dirs, cfg, tmp_path):
    _, tgt_dir = synthetic_dirs
    target = _clean(tgt_dir, cfg)
    cols = numeric_feature_columns(target, "Label")
    target_split = build_splits(target, cols, "Label", cfg["data"], 0)
    source_like = build_splits(
        target, cols, "Label", cfg["data"], 1,
        preprocessor=target_split["preprocessor"],
        scaler_fit_dataset="source",
        scaler_fit_split="train",
        preprocessing_strategy="source_scaler_for_transfer",
    )
    target_split["X_transfer_train"] = source_like["X_train"]
    target_split["X_transfer_val"] = source_like["X_val"]
    target_split["X_transfer_test"] = source_like["X_test"]
    target_split["transfer_preprocessor"] = source_like["preprocessor"].to_dict()
    target_split["preprocessing_metadata"]["transfer"] = source_like["preprocessing_metadata"]
    save_splits(tmp_path, "target", target_split)
    loaded = load_splits(tmp_path, "target")
    assert loaded["X_transfer_train"].shape == loaded["X_train"].shape
    assert loaded["transfer_preprocessor"]["feature_columns"] == cols
    assert loaded["preprocessing_metadata"]["transfer"]["scaler_fit_dataset"] == "source"
    assert np.isfinite(loaded["X_transfer_train"]).all()


def test_streamlit_input_alignment_accepts_original_target_names():
    from ddos_tl.data.features import align_features

    alignment = align_features(
        ["Total Length of Fwd Packets", "Flow Duration"],
        ["Fwd Packets Length Total", "Flow Duration"],
    )
    raw_target = pd.DataFrame({
        "Fwd Packets Length Total": [1.0],
        "Flow Duration": [2.0],
    })
    aligned = apply_alignment(raw_target, alignment, "target")
    assert "fwd_packets_length_total" in aligned.columns


def test_by_file_split_validates_files_and_classes():
    df = pd.DataFrame({
        "_file": ["train", "train", "test", "test"],
        "Label": ["A", "B", "A", "B"],
        "x": [1, 2, 3, 4],
    })
    train, val, test = by_file_split(df, "Label", ["train"], ["test"], 0.5, 0)
    assert not train.empty and not val.empty and not test.empty
    with pytest.raises(ValueError, match="inconnu"):
        by_file_split(df, "Label", ["missing"], ["test"], 0.5, 0)
    with pytest.raises(ValueError, match="train et test"):
        by_file_split(df, "Label", ["train"], ["train"], 0.5, 0)


def test_cleaning_reports_nan_inf_constant_invalid_and_duplicates():
    df = pd.DataFrame({
        "Label": ["A", "A", "B"],
        "value": [1.0, 1.0, np.nan],
        "positive": [np.inf, np.inf, 2.0],
        "negative": [-np.inf, -np.inf, 3.0],
        "constant": [7.0, 7.0, 7.0],
        "invalid": ["bad", "bad", "bad"],
    })
    cleaned, stats = clean_frame(df, "Label", [], return_stats=True)
    assert stats["positive_inf_by_column_before"]["positive"] == 2
    assert stats["negative_inf_by_column_before"]["negative"] == 2
    assert stats["entirely_invalid_columns"] == ["invalid"]
    assert "constant" in stats["constant_columns"]
    assert stats["duplicate_rows_removed"] == 1
    assert "invalid" not in cleaned.columns
    assert cleaned["value"].isna().sum() == 1


def test_pipeline_rejects_forbidden_or_misaligned_features():
    with pytest.raises(ValueError, match="interdite"):
        validate_feature_alignment(["Flow Duration", "Label"])
    with pytest.raises(ValueError, match="ordre"):
        validate_feature_alignment(
            ["b", "a"], {"common": ["a", "b"]}
        )


def test_split_overlap_control_and_metadata(synthetic_dirs, cfg):
    _, tgt_dir = synthetic_dirs
    df = _clean(tgt_dir, cfg)
    parts = (df.iloc[:10], df.iloc[10:20], df.iloc[20:30])
    assert validate_split_disjointness(parts)["overlap_counts"] == {
        "train_val": 0, "train_test": 0, "val_test": 0
    }
    with pytest.raises(ValueError, match="Chevauchement"):
        validate_split_disjointness((df.iloc[:10], df.iloc[5:15], df.iloc[20:30]))
    split = build_splits(df, numeric_feature_columns(df, "Label"), "Label", cfg["data"], 0)
    assert split["split_metadata"]["preprocessor_fit_split"] == "train"
    assert all(value == 0 for value in split["split_metadata"]["overlap_counts"].values())


def test_report_markdown_contains_audit_sections():
    markdown = report_markdown({
        "provenance": {"source": "test"},
        "memory_preflight": {"input_bytes": 1, "estimated_peak_bytes": 3,
                             "sufficient_for_estimate": True},
        "target": {"rows_after_cleaning": 2, "class_counts": {"A": 2},
                   "cleaning": {}, "split_counts": {}},
        "leakage_controls": {"scaler_fit": "train split only"},
    })
    assert "# Data preparation report" in markdown
    assert "## Provenance" in markdown
    assert "scaler_fit" in markdown
