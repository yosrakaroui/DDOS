import numpy as np
import torch

from ddos_tl.experiments.two_by_two import (
    build_grid,
    build_initialization_bundle,
    cell_inputs,
    instantiate_model,
    state_hashes,
    validate_grid,
)
from ddos_tl.training.loops import make_loader
from ddos_tl.models.mlp import TabularMLP


def test_grid_has_exactly_48_unique_configurations():
    grid = build_grid()
    validate_grid(grid)
    assert len(grid) == 48
    assert len({row["run_id"] for row in grid}) == 48


def test_shared_initial_states_and_head(tmp_path):
    source = TabularMLP(76, 2, hidden=(16, 8), dropout=0.0)
    path = tmp_path / "backbone.pt"
    torch.save(source.backbone_state_dict(), path)
    a = build_initialization_bundle(1, 76, (16, 8), 0.0, path)
    b = build_initialization_bundle(1, 76, (16, 8), 0.0, path)
    assert state_hashes(a) == state_hashes(b)
    random_a = instantiate_model(a, "random", 76, (16, 8), 0.0)
    random_c = instantiate_model(a, "random", 76, (16, 8), 0.0)
    pretrained_b = instantiate_model(a, "pretrained", 76, (16, 8), 0.0)
    pretrained_d = instantiate_model(a, "pretrained", 76, (16, 8), 0.0)
    assert all(torch.equal(random_a.state_dict()[k], random_c.state_dict()[k]) for k in random_a.state_dict())
    assert all(torch.equal(pretrained_b.state_dict()[k], pretrained_d.state_dict()[k]) for k in pretrained_b.state_dict())
    assert all(torch.equal(random_a.head.state_dict()[k], pretrained_b.head.state_dict()[k]) for k in random_a.head.state_dict())
    assert random_a.head.out_features == 18


def test_cells_share_subset_indices_and_labels():
    y = np.repeat(np.arange(3), 20)
    data = {
        "X_train": np.arange(60 * 4, dtype=np.float32).reshape(60, 4),
        "X_transfer_train": np.arange(60 * 4, dtype=np.float32).reshape(60, 4) + 1,
        "X_val": np.zeros((3, 4), dtype=np.float32),
        "X_transfer_val": np.ones((3, 4), dtype=np.float32),
        "X_test": np.zeros((3, 4), dtype=np.float32),
        "X_transfer_test": np.ones((3, 4), dtype=np.float32),
        "y_train": y,
        "y_val": np.array([0, 1, 2]),
        "y_test": np.array([0, 1, 2]),
    }
    target = cell_inputs(data, "target", 0.1, 2)
    source = cell_inputs(data, "source", 0.1, 2)
    assert np.array_equal(target["subset_indices"], source["subset_indices"])
    assert np.array_equal(target["y_train"], source["y_train"])
    assert target["X_val"].shape == source["X_val"].shape


def test_dataloader_generator_order_is_shared_for_same_seed():
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.arange(20, dtype=np.int64)
    first = make_loader(X, y, batch_size=4, shuffle=True, seed=7)
    second = make_loader(X, y, batch_size=4, shuffle=True, seed=7)
    first_order = np.concatenate([batch_y.numpy() for _, batch_y in first])
    second_order = np.concatenate([batch_y.numpy() for _, batch_y in second])
    assert np.array_equal(first_order, second_order)
