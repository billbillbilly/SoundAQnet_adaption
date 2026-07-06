"""PyTorch models for Emo-Soundscapes valence/arousal inference."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmoSoundscapeCNN(nn.Module):
    """CNN described by Fan et al. (2018) for 54 x 30 feature patches.

    The paper trains separate one-output models for valence and arousal.  This
    implementation also supports a two-output checkpoint through ``out_dim=2``.
    """

    def __init__(self, out_dim: int = 1, dropout: float = 0.15) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(3, 3), stride=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(8 * 11 * 5, 256)
        self.fc_out = nn.Linear(256, out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a batch of 54 x 30 feature patches.

        Accepts ``(B, 54, 30)`` or ``(B, 1, 54, 30)`` tensors.
        """
        if features.ndim == 3:
            features = features.unsqueeze(1)
        if features.ndim != 4:
            raise ValueError("Expected features with shape (B, 54, 30) or (B, 1, 54, 30)")

        x = F.relu(self.conv1(features))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc_out(x)
