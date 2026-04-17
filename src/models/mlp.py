"""
MLP regressor head on top of ESM-2 embeddings.
Input: 1280-dim ESM-2 mean-pooled embedding
Output: scalar aggrescan3d_avg_value prediction
"""

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Feed-forward MLP with BatchNorm, GELU activations and Dropout.

    Architecture (defaults):
        1280 → 512 → 256 → 128 → 1
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
