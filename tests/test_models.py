import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ddos_tl.models.mlp import TabularMLP
from ddos_tl.training.strategies import STRATEGIES, run_strategy


def test_head_replacement_and_forward_shape():
    m = TabularMLP(20, 3, hidden=(16, 8))
    m.replace_head(5)
    assert m(torch.zeros(4, 20)).shape == (4, 5)


def test_freezing_blocks():
    m = TabularMLP(20, 3, hidden=(16, 8, 4))
    m.freeze_blocks(2)
    flags = [all(p.requires_grad for p in b.parameters()) for b in m.blocks]
    assert flags == [False, False, True]
    assert all(p.requires_grad for p in m.head.parameters())
    m.freeze_blocks(len(m.blocks))
    assert m.trainable_parameters() == sum(p.numel() for p in m.head.parameters())
    m.unfreeze_all()
    assert m.trainable_parameters() == sum(p.numel() for p in m.parameters())


def test_backbone_transfer_copies_weights_but_not_head():
    a, b = TabularMLP(20, 3, hidden=(16, 8)), TabularMLP(20, 4, hidden=(16, 8))
    b.load_backbone(a.backbone_state_dict())
    assert torch.equal(a.blocks[0][0].weight, b.blocks[0][0].weight)
    assert a.head.out_features != b.head.out_features


def _toy_data(n=600, d=20, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (k, d))
    y = rng.integers(0, k, n)
    X = (centers[y] + rng.normal(0, 1, (n, d))).astype(np.float32)
    h = n // 5
    return {"X_train": X[: 3 * h], "y_train": y[: 3 * h], "X_val": X[3 * h: 4 * h], "y_val": y[3 * h: 4 * h],
            "X_test": X[4 * h:], "y_test": y[4 * h:], "class_names": [f"c{i}" for i in range(k)]}


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_runs_and_learns(strategy, cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg["training"].update(lr=0.01, epochs=15, patience=15)
    data = _toy_data()
    pre = None
    if strategy != "scratch":
        src = TabularMLP(20, 3, cfg["model"]["hidden"], cfg["model"]["dropout"])
        pre = tmp_path / "pre.pt"
        torch.save(src.backbone_state_dict(), pre)
    res = run_strategy(strategy, data, cfg, seed=0, pretrained_path=pre, device=torch.device("cpu"))
    assert res["test_metrics"]["macro_f1"] > 0.5  # hasard = 0.33 pour 3 classes
    assert res["val_macro_f1"] == pytest.approx(
        max(row["val_macro_f1"] for row in res["history"])
    )
    assert res["best_epoch"] in {row["epoch"] for row in res["history"]}
