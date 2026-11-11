"""Deterministic, non-training specification for the scaler/initialization 2x2."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ..data.splitting import subsample_indices
from ..models.mlp import TabularMLP
from ..utils.seed import set_seed

SCALERS = ("target", "source")
INITIALIZATIONS = ("random", "pretrained")
FRACTIONS = (0.01, 0.05, 0.10, 1.0)
SEEDS = (0, 1, 2)
N_CLASSES = 18


@dataclass(frozen=True)
class Cell:
    initialization: Literal["random", "pretrained"]
    scaler: Literal["target", "source"]

    @property
    def name(self) -> str:
        return f"{self.initialization}_{self.scaler}"


@dataclass
class InitializationBundle:
    seed: int
    random_state: dict[str, torch.Tensor]
    pretrained_backbone: dict[str, torch.Tensor]


def build_grid(
    fractions: tuple[float, ...] = FRACTIONS,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict]:
    """Build the complete 2x2 x fraction x seed manifest without training."""
    cells = [
        Cell(initialization=initialization, scaler=scaler)
        for initialization in INITIALIZATIONS
        for scaler in SCALERS
    ]
    return [
        {
            "run_id": f"{cell.name}_f{fraction}_s{seed}",
            "cell": cell.name,
            "initialization": cell.initialization,
            "scaler": cell.scaler,
            "fraction": fraction,
            "seed": seed,
        }
        for cell in cells
        for fraction in fractions
        for seed in seeds
    ]


def validate_grid(grid: list[dict], expected_count: int = 48) -> None:
    """Validate uniqueness and the exact factorial design."""
    expected_cells = {
        f"{initialization}_{scaler}"
        for initialization in INITIALIZATIONS
        for scaler in SCALERS
    }
    keys = {(row["cell"], float(row["fraction"]), int(row["seed"])) for row in grid}
    if len(grid) != expected_count or len(keys) != expected_count:
        raise ValueError(f"Expected {expected_count} unique 2x2 runs, got {len(grid)}")
    if {row["cell"] for row in grid} != expected_cells:
        raise ValueError("The grid does not contain all four 2x2 cells")
    if {float(row["fraction"]) for row in grid} != set(FRACTIONS):
        raise ValueError("Unexpected fractions in the 2x2 grid")
    if {int(row["seed"]) for row in grid} != set(SEEDS):
        raise ValueError("Unexpected seeds in the 2x2 grid")


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_initialization_bundle(
    seed: int,
    in_dim: int,
    hidden: tuple[int, ...] | list[int],
    dropout: float,
    pretrained_path: str | Path,
) -> InitializationBundle:
    """Create shared random/head state and load an untouched pretrained backbone."""
    set_seed(seed)
    random_model = TabularMLP(in_dim, N_CLASSES, hidden, dropout)
    random_state = _clone_state(random_model.state_dict())
    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=True)
    pretrained_backbone = _clone_state(pretrained)
    expected = set(random_model.backbone_state_dict())
    if set(pretrained_backbone) != expected:
        raise ValueError("Pretrained backbone keys do not match the target architecture")
    return InitializationBundle(seed, random_state, pretrained_backbone)


def instantiate_model(
    bundle: InitializationBundle,
    initialization: Literal["random", "pretrained"],
    in_dim: int,
    hidden: tuple[int, ...] | list[int],
    dropout: float,
) -> TabularMLP:
    """Instantiate a target model with a shared head and selected backbone."""
    model = TabularMLP(in_dim, N_CLASSES, hidden, dropout)
    state = _clone_state(bundle.random_state)
    if initialization == "pretrained":
        state.update(_clone_state(bundle.pretrained_backbone))
    elif initialization != "random":
        raise ValueError(f"Unknown initialization: {initialization}")
    model.load_state_dict(state, strict=True)
    return model


def state_hashes(bundle: InitializationBundle) -> dict[str, str]:
    """Return hashes useful for proving shared initial states."""
    return {
        "random_full": _state_hash(bundle.random_state),
        "random_backbone": _state_hash({
            key: value for key, value in bundle.random_state.items()
            if key.startswith("blocks.")
        }),
        "shared_target_head": _state_hash({
            key: value for key, value in bundle.random_state.items()
            if key.startswith("head.")
        }),
        "pretrained_backbone": _state_hash(bundle.pretrained_backbone),
    }


def cell_inputs(data: dict, scaler: Literal["target", "source"], fraction: float, seed: int) -> dict:
    """Select the scaler variant and identical train subset for one cell."""
    prefix = "X_transfer_" if scaler == "source" else "X_"
    if scaler == "source" and "X_transfer_train" not in data:
        raise ValueError("Source-scaled target artifacts are missing")
    indices = subsample_indices(data["y_train"], fraction, seed)
    return {
        "X_train": data[f"{prefix}train"][indices],
        "y_train": data["y_train"][indices],
        "X_val": data[f"{prefix}val"],
        "y_val": data["y_val"],
        "X_test": data[f"{prefix}test"],
        "y_test": data["y_test"],
        "subset_indices": indices,
        "scaler": scaler,
    }


def clipping_rates(standardized: np.ndarray, clip: float = 10.0) -> np.ndarray:
    """Compute per-feature clipping rates from pre-clipped standardized values."""
    if standardized.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    return np.mean(np.abs(standardized) > clip, axis=0)
