"""Residual CNN for spectrogram-like inputs.

Assumes input shape (B, F, T) or (B, 1, F, T). We standardize to (B, 1, F, T).
Future Enhancements:
- Squeeze-and-Excitation blocks.
- Multi-scale temporal dilations.
- Export-friendly forward for ONNX / edge.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if in_ch != out_ch or stride != 1:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out += identity
        out = self.relu(out)
        return out

class ResNetSpectrogram(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, num_classes: int = 8, depth: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        ch = base_channels
        blocks = []
        for d in range(depth):
            blocks.append(BasicBlock(ch, ch))
            if d % 2 == 1:  # Downsample every 2 blocks
                blocks.append(BasicBlock(ch, ch * 2, stride=2))
                ch *= 2
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, F, T) -> reshape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,F,T)
        out = self.stem(x)
        out = self.blocks(out)
        out = self.pool(out).flatten(1)
        return self.head(out)
