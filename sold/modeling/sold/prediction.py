from modeling.attention_mask import AttentionMask, CausalMask, AlibiMask
from modeling.distributions import TwoHotEncodingDistribution
from modeling.positional_encoding import PositionalEncoding, NoPositionalEncoding, LearnedEncoding, SinusoidalEncoding
from .slot_aggregation_transformer import SlotAggregationTransformer
import torch
import torch.distributions as D
import torch.nn as nn
from typing import Optional, List
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Predictor(nn.Module):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, token_dim: int, num_heads: int,
                 num_layers: int, hidden_dim: int, output_dim: int, num_register_tokens: int = 0,
                 num_mlp_layers: int = 1, selected_slot_indices: Optional[List[int]] = None) -> None:
        """Used to predict single quantities like rewards or actions from the set and history of slots."""
        super().__init__()

        self.selected_slot_indices = selected_slot_indices
        effective_num_slots = len(selected_slot_indices) if selected_slot_indices else num_slots

        # attention_mask = AlibiMask(max_episode_steps, num_heads, num_slots, num_register_tokens)
        attention_mask = AlibiMask(max_episode_steps, num_heads, effective_num_slots, num_register_tokens)
        positional_encoding = NoPositionalEncoding(max_episode_steps, token_dim)

        #attention_mask = CausalMask(max_episode_steps, num_heads, num_slots, num_register_tokens)
        #positional_encoding = SinusoidalEncoding(max_episode_steps, token_dim)
        #positional_encoding = LearnedEncoding(max_episode_steps, token_dim)
        # self.slot_aggregation_transformer = SlotAggregationTransformer(
        #     attention_mask, positional_encoding, max_episode_steps, num_slots, slot_dim, token_dim, num_heads,
        #     num_layers, hidden_dim, num_register_tokens)
        self.slot_aggregation_transformer = SlotAggregationTransformer(
            attention_mask, positional_encoding, max_episode_steps, effective_num_slots, slot_dim, token_dim, num_heads,
            num_layers, hidden_dim, num_register_tokens)

        self.mlp = []
        for layer_num in range(num_mlp_layers):
            if layer_num == 0:
                self.mlp.append(nn.Linear(token_dim, hidden_dim))
            else:
                self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(RMSNorm(hidden_dim))
            self.mlp.append(nn.SiLU())
        self.mlp.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, slots: torch.Tensor, start: int = 0) -> torch.Tensor:
        if self.selected_slot_indices is not None:
            slots = slots[:, :, self.selected_slot_indices, :]
        
        features = self.slot_aggregation_transformer(slots, start=start)
        out = self.mlp(features)
        return out


class ScalarPredictor(Predictor):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, token_dim: int, num_heads: int,
                 num_layers: int, hidden_dim: int, num_register_tokens: int = 0, num_mlp_layers: int = 1,
                 selected_slot_indices: Optional[List[int]] = None) -> None:
        """Predict scalar outputs like values and rewards directly"""
        super().__init__(max_episode_steps, num_slots, slot_dim, token_dim, num_heads, num_layers, hidden_dim,
                         output_dim=1, num_register_tokens=num_register_tokens, num_mlp_layers=num_mlp_layers,
                         selected_slot_indices=selected_slot_indices)

    def forward(self, slots: torch.Tensor, start: int = 0) -> D.Distribution:
        loc = super().forward(slots, start=start)
        return D.Normal(loc, 1.0, validate_args=False)


class GaussianPredictor(Predictor):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, token_dim: int, num_heads: int,
                 num_layers: int, hidden_dim: int, output_dim: int, num_register_tokens: int = 0, num_mlp_layers: int = 1,
                 lower_bound = None, upper_bound = None, selected_slot_indices: Optional[List[int]] = None) -> None:
        super().__init__(max_episode_steps, num_slots, slot_dim, token_dim, num_heads, num_layers, hidden_dim,
                         output_dim=2*output_dim, num_register_tokens=num_register_tokens, num_mlp_layers=num_mlp_layers,
                         selected_slot_indices=selected_slot_indices)
        self.max_std, self.min_std, self.init_std = 1.0, 0.1, 2.0
        self.lower_bound = torch.tensor(lower_bound)
        self.upper_bound = torch.tensor(upper_bound)

    def forward(self, slots: torch.Tensor, start: int = 0) -> D.Distribution:
        x = super().forward(slots, start=start)
        mean_, std_ = x.chunk(2, -1)
        mean_ = torch.clamp(mean_, self.lower_bound.to(mean_.device), self.upper_bound.to(mean_.device))
        std = (self.max_std - self.min_std) * torch.sigmoid(std_ + self.init_std) + self.min_std
        dist = D.Normal(torch.tanh(mean_), std, validate_args=False)
        return D.Independent(dist, 1, validate_args=False)


class TwoHotPredictor(Predictor):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, token_dim: int, num_heads: int,
                 num_layers: int, hidden_dim: int, num_register_tokens: int = 0, num_mlp_layers: int = 1,
                 selected_slot_indices: Optional[List[int]] = None) -> None:
        """Predict over 255 exponentially-spaced bins to represent scalar values like rewards."""
        super().__init__(max_episode_steps, num_slots, slot_dim, token_dim, num_heads, num_layers, hidden_dim,
                         output_dim=255, num_register_tokens=num_register_tokens, num_mlp_layers=num_mlp_layers,
                         selected_slot_indices=selected_slot_indices)

    def forward(self, slots: torch.Tensor, start: int = 0) -> TwoHotEncodingDistribution:
        logits = super().forward(slots, start=start)
        return TwoHotEncodingDistribution(logits, dims=1)


class MLPPredictor(nn.Module):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, 
                 hidden_dim: int, output_dim: int, num_mlp_layers: int = 1) -> None:
        super().__init__()

        self.mlp = []
        for layer_num in range(num_mlp_layers):
            if layer_num == 0:
                self.mlp.append(nn.Linear(slot_dim * num_slots, hidden_dim))
            else:
                self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.LayerNorm(hidden_dim))
            self.mlp.append(nn.SiLU())
        self.mlp.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, slots: torch.Tensor, start: int = 0) -> torch.Tensor:
        t = slots.shape[1]
        slots = slots[:, -(t-start):]
        out = self.mlp(rearrange(slots, 'b t n d -> b t (n d)'))
        return out


class GaussianMLPPredictor(MLPPredictor):
    """MLP-based actor that only uses the current timestep."""
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, 
                 hidden_dim: int, output_dim: int, num_mlp_layers: int = 1,
                 lower_bound = None, upper_bound = None, ) -> None:
        super().__init__(max_episode_steps, num_slots, slot_dim, hidden_dim, output_dim=2*output_dim, num_mlp_layers=num_mlp_layers)

        self.max_std, self.min_std, self.init_std = 1.0, 0.1, 2.0
        self.lower_bound = torch.tensor(lower_bound)
        self.upper_bound = torch.tensor(upper_bound)

    def forward(self, slots: torch.Tensor, start: int = 0) -> D.Distribution:
        x = super().forward(slots, start=start)
        mean_, std_ = x.chunk(2, -1)
        mean_ = torch.clamp(mean_, self.lower_bound.to(mean_.device), self.upper_bound.to(mean_.device))
        std = (self.max_std - self.min_std) * torch.sigmoid(std_ + self.init_std) + self.min_std
        dist = D.Normal(torch.tanh(mean_), std, validate_args=False)
        return D.Independent(dist, 1, validate_args=False)


class TwoHotMLPPredictor(MLPPredictor):
    def __init__(self, max_episode_steps: int, num_slots: int, slot_dim: int, 
                 hidden_dim: int, output_dim: int, num_mlp_layers: int = 1) -> None:
        super().__init__(max_episode_steps, num_slots, slot_dim, hidden_dim, output_dim=255, num_mlp_layers=num_mlp_layers)

    def forward(self, slots: torch.Tensor, start: int = 0) -> TwoHotEncodingDistribution:
        logits = super().forward(slots, start=start)
        return TwoHotEncodingDistribution(logits, dims=1)