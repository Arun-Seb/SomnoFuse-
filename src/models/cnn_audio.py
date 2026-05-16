"""
src/models/cnn_audio.py

1-D temporal CNN and 2-D spectrogram CNN for snoring event detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_ch: int, out_ch: int,
        kernel: int, stride: int = 1,
        padding: int = 0, groups: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ──────────────────────────────────────────────────────────────────────────────
# 2-D Mel-spectrogram CNN
# ──────────────────────────────────────────────────────────────────────────────

class SnoreCNN(nn.Module):
    """
    2-D CNN operating on log-Mel spectrograms of the Schnarc channel.

    Input  : (batch, 1, n_mels, n_frames)
    Output : (batch, embed_dim)  – embedding, or (batch, n_classes) if classify=True
    """

    def __init__(
        self,
        n_mels: int = 64,
        embed_dim: int = 256,
        n_classes: int = 9,
        dropout: float = 0.3,
        classify: bool = False,
    ):
        super().__init__()
        self.classify = classify

        self.encoder = nn.Sequential(
            # Block 1 – 1 × n_mels × T
            ConvBNReLU(1, 32, kernel=3, padding=1),
            ConvBNReLU(32, 32, kernel=3, padding=1),
            nn.MaxPool2d(2, 2),          # → 32 × (n_mels/2) × (T/2)
            nn.Dropout2d(0.1),

            # Block 2
            ConvBNReLU(32, 64, kernel=3, padding=1),
            SEBlock(64),
            ConvBNReLU(64, 64, kernel=3, padding=1),
            nn.MaxPool2d(2, 2),          # → 64 × (n_mels/4) × (T/4)
            nn.Dropout2d(0.1),

            # Block 3
            ConvBNReLU(64, 128, kernel=3, padding=1),
            SEBlock(128),
            ConvBNReLU(128, 128, kernel=3, padding=1),
            nn.MaxPool2d(2, 2),          # → 128 × (n_mels/8) × (T/8)
            nn.Dropout2d(0.2),

            # Block 4
            ConvBNReLU(128, 256, kernel=3, padding=1),
            SEBlock(256),
            nn.AdaptiveAvgPool2d((1, 1)),  # → 256 × 1 × 1
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(embed_dim, n_classes) if classify else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, n_mels, T)
        """
        feat = self.encoder(x)
        emb  = self.head(feat)
        return self.classifier(emb)


# ──────────────────────────────────────────────────────────────────────────────
# 1-D Temporal CNN for raw waveform
# ──────────────────────────────────────────────────────────────────────────────

class SnoreWaveformCNN(nn.Module):
    """
    1-D dilated temporal CNN for the raw 256 Hz snoring waveform.

    Input  : (batch, 1, n_samples)
    Output : (batch, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        base_channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        def _block(in_ch: int, out_ch: int, dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=7, dilation=dilation,
                          padding=3 * dilation, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        c = base_channels
        self.layers = nn.Sequential(
            _block(1, c,     dilation=1),
            _block(c, c * 2, dilation=2),
            _block(c * 2, c * 4, dilation=4),
            _block(c * 4, c * 8, dilation=8),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(c * 8, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, T)"""
        out = self.layers(x).squeeze(-1)  # (B, C)
        return self.proj(out)
