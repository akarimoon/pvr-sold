import torch
import torch.nn as nn
from typing import Optional
import torch.distributions as D

from modeling.distributions import TwoHotEncodingDistribution
from modeling.dreamer.utils import trunc_xavier_normal_weight_init, xavier_uniform_weight_init

class Predictor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 norm: bool = True) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim

        self.mlp.apply(trunc_xavier_normal_weight_init)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        b, t, e = states.shape
        return self.mlp(states.view(b * t, e)).view(b, t, self.output_dim)


class GaussianPredictor(Predictor):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 norm: bool = True) -> None:
        super().__init__(input_dim, 2 * output_dim, hidden_dim, num_layers, norm)
        self.max_std, self.min_std, self.init_std = 1.0, 0.1, 2.0

        self.mlp[-1].apply(xavier_uniform_weight_init(1.0))

    def forward(self, states: torch.Tensor) -> torch.distributions.Distribution:
        x = super().forward(states)  # (B, T, 2*D)
        mean_, std_ = x.chunk(2, dim=-1)
        std = (self.max_std - self.min_std) * torch.sigmoid(std_ + self.init_std) + self.min_std
        dist = D.Normal(torch.tanh(mean_), std, validate_args=False)
        return D.Independent(dist, 1, validate_args=False)


class TwoHotPredictor(Predictor):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int = 255,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 norm: bool = True) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, num_layers, norm)

        self.mlp[-1].apply(xavier_uniform_weight_init(0.0))

    def forward(self, states: torch.Tensor) -> TwoHotEncodingDistribution:
        logits = super().forward(states)
        return TwoHotEncodingDistribution(logits, dims=1)


