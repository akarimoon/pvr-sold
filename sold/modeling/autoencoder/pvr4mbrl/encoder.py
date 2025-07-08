from typing import List, Tuple
import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    """Just a MLP Encoder."""

    def __init__(self, vit_feature_dim: int, hidden_dims: List[int], feature_dim: int) -> None:
        super().__init__()

        layers = []
        layers.append(nn.LayerNorm(vit_feature_dim))
        inp_dim = vit_feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(inp_dim, hidden_dim))
            layers.append(nn.ReLU())
            inp_dim = hidden_dim
        layers.append(nn.Linear(inp_dim, feature_dim))
        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, vit_features: torch.Tensor) -> torch.Tensor:
        embeddings = self.shared_mlp(vit_features)
        return embeddings
