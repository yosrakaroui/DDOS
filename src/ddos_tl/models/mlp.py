"""MLP tabulaire avec séparation nette backbone / tête (nécessaire au Transfer Learning)."""
from __future__ import annotations

import torch
from torch import nn


class TabularMLP(nn.Module):
    """blocks = backbone (Linear -> ReLU -> Dropout) ; head = couche de classification."""

    def __init__(self, in_dim: int, n_classes: int, hidden=(256, 128, 64), dropout: float = 0.2):
        super().__init__()
        hidden = list(hidden)
        dims = [in_dim, *hidden]
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout))
            for i in range(len(hidden))
        )
        self.head = nn.Linear(hidden[-1], n_classes)
        self.arch = {"in_dim": in_dim, "hidden": hidden, "dropout": dropout}

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    # ---- Transfer Learning -------------------------------------------------
    def backbone_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k.startswith("blocks.")}

    def load_backbone(self, state: dict) -> None:
        """Charge uniquement les poids du backbone (la tête reste initialisée aléatoirement)."""
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"Clés inattendues dans le backbone pré-entraîné : {result.unexpected_keys}")
        missing = [k for k in result.missing_keys if k.startswith("blocks.")]
        if missing:
            raise ValueError(f"Backbone incomplet, clés manquantes : {missing}")

    def replace_head(self, n_classes: int) -> None:
        """Remplace la couche de classification (nouvelle initialisation)."""
        self.head = nn.Linear(self.head.in_features, n_classes)

    # ---- Freezing ----------------------------------------------------------
    def freeze_blocks(self, k: int) -> None:
        """Gèle les k premiers blocs du backbone (k = len(blocks) -> backbone entièrement gelé)."""
        for i, block in enumerate(self.blocks):
            for p in block.parameters():
                p.requires_grad = i >= k

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
