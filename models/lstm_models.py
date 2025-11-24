"""LSTM-based classifier for sequential feature frames.

Input shape expectation: (B, F, T) where F=feature dimension, T=time.
We transpose to (B, T, F) for nn.LSTM.
Future Enhancements:
- Bidirectional toggle.
- Attention pooling / self-attention blocks.
- Distillation hook to expose intermediate states.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int = 120, hidden_dim: int = 128, num_layers: int = 2, bidirectional: bool = True, dropout: float = 0.3, num_classes: int = 8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, F, T) -> (B, T, F)
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        packed_out, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]
        return self.classifier(last)
