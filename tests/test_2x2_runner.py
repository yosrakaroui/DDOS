import pytest

torch = pytest.importorskip("torch")

from ddos_tl.config import load_config
from ddos_tl.experiments.runner import OUTPUT_DIR, dry_run, load_manifest, run_one, validate_all


def test_runner_uses_fixed_output_and_manifest_has_48_runs():
    manifest = load_manifest()
    assert OUTPUT_DIR.as_posix().endswith("reports/results/phase_2x2_scaler_initialization")
    assert len(manifest["grid"]) == 48
    assert {row["cell"] for row in manifest["grid"]} == {
        "random_target", "pretrained_target", "random_source", "pretrained_source"
    }


def test_dry_run_does_not_call_fit(monkeypatch):
    import ddos_tl.experiments.runner as runner

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fit must not be called in dry-run")

    monkeypatch.setattr(runner, "fit", fail_if_called)
    cfg = load_config("configs/default.yaml")
    manifest = load_manifest()
    checks = dry_run(cfg, manifest)
    assert len(checks) == 48


def test_validate_all_has_common_hyperparameters_and_subsets():
    cfg = load_config("configs/default.yaml")
    manifest = load_manifest()
    checks = validate_all(cfg, manifest)
    assert len(checks) == 48
    assert manifest["shared_training_protocol"]["optimizer"] == "Adam"
    assert manifest["shared_training_protocol"]["dataloader_generator_seed"] == (
        "same experiment seed for every cell"
    )


def test_manifest_loader_rejects_alternate_path(tmp_path):
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "other.json")


def test_runner_refuses_to_overwrite_existing_result():
    manifest = load_manifest()
    row = manifest["grid"][0]
    result_path = OUTPUT_DIR / f"{row['run_id']}.json"
    result_path.write_text("sentinel", encoding="utf-8")
    try:
        with pytest.raises(FileExistsError):
            run_one(row, {}, load_config("configs/default.yaml"), manifest, torch.device("cpu"))
    finally:
        result_path.unlink()
