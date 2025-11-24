"""Fully-connected classifier heads.

Intended for flattened statistical features (e.g., mean MFCC, pooled embeddings).
Design:
- SimpleFC: optional dropout & batchnorm.
Future:
- Projection + multi-task heads (emotion + intensity).
- Knowledge distillation student (smaller hidden sizes).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

class SimpleFC(nn.Module):
    def __init__(self, input_dim: int = 120, num_classes: int = 8, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        layers = []
        dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x shape (B, F)
        return self.classifier(self.backbone(x))
