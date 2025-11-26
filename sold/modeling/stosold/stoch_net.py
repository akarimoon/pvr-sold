from typing import Union, Tuple, Dict
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from modeling.sold.slot_aggregation_transformer import TransformerEncoderLayer

class BaseStochNet(nn.Module):
    def __init__(self, input_dim, num_slots, output_dim, slot_dim = None, initial: str = "zeros"):
        super().__init__()
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.output_dim = output_dim
        self.slot_dim = slot_dim
        self._initial = initial

        if slot_dim is not None:
            self.stoch_mean = nn.Linear(input_dim + num_slots * slot_dim, output_dim)
            self.stoch_sigma = nn.Linear(input_dim + num_slots * slot_dim, output_dim)
        else:
            self.stoch_mean = nn.Linear(input_dim, output_dim)
            self.stoch_sigma = nn.Linear(input_dim, output_dim)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._initial == "zeros":
            stoch = torch.zeros(batch_size, self.output_dim, device=device)
        elif self._initial == "gaussian":
            stoch = torch.randn(batch_size, self.output_dim, device=device)
        else:
            raise ValueError(f"Invalid initial state: {self._initial}")
        return stoch.unsqueeze(1)

    def get_dist(self, mean: torch.Tensor, std: torch.Tensor) -> D.Normal:
        return D.Normal(mean, std)

    def sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mean + std * torch.randn_like(mean).to(mean.device)
        else:
            return mean

    def forward(self, x: torch.Tensor, slots: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if slots is not None:
            slots = rearrange(slots, "b t k d -> b t (k d)")
            x = torch.cat([x, slots], dim=-1)
        mean = self.stoch_mean(x)
        std = F.softplus(self.stoch_sigma(x)) + 1e-6
        return mean, std


class StochPrior(BaseStochNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.stoch_mean(x)
        std = F.softplus(self.stoch_sigma(x)) + 1e-6
        return mean, std


class StochPosterior(BaseStochNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def init(self, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    #     slots = torch.cat([slots[:, :1], slots], dim=1)
    #     x = torch.cat([x[:, :1], x], dim=1)[:, :-1]
    #     return x, slots

    def forward(self, x: torch.Tensor, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            x: (B, T+1, E)
            slots: (B, T+1, D)
        Outputs:
            mean: (B, T, O)
            std: (B, T, O)
        """
        slots = rearrange(slots, "b t k d -> b t (k d)")
        slots = slots[:, 1:] - slots[:, :-1]
        x = torch.cat([x[:, :-1], slots], dim=-1)
        mean = self.stoch_mean(x)
        std = F.softplus(self.stoch_sigma(x)) + 1e-6
        return mean, std


class StochNetBackbone(nn.Module):
    def __init__(self, slot_dim, token_dim, num_heads, num_layers, mlp_dim, dropout):
        super().__init__()
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.slot_projection = nn.Linear(slot_dim, self.token_dim)
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.transformer = nn.Sequential(
            *[TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=self.num_heads,
                dim_feedforward=self.mlp_dim,
                dropout=self.dropout,
            ) for _ in range(self.num_layers)]
        )

    def forward(self, slots: torch.Tensor, start: int = 0) -> torch.Tensor:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        slots = self.slot_projection(slots)
        act_tokens = self.act_token.repeat(batch_size, sequence_length, 1, 1)
        slots = torch.cat([act_tokens, slots], dim=2)
        slots = rearrange(slots, "b t k d -> (b t) k d")
        slots = self.transformer(slots)
        slots = rearrange(slots, "(b t) k d -> b t k d", b=batch_size, t=sequence_length)
        slots = slots[:, -(sequence_length-start):, 0]
        return slots

class StochNet(nn.Module):
    def __init__(self, input_dim, num_slots, output_dim, slot_dim = None, action_dim = None, share_backbone: bool = True,
                 use_action: bool = False, initial: str = "zeros", bound_mean_std: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.output_dim = output_dim
        self.slot_dim = slot_dim
        self.action_dim = action_dim
        self.share_backbone = share_backbone
        self.use_action = use_action
        self._initial = initial
        self._bound_mean_std = bound_mean_std

        # if self._initial == "learned":
        #     self.W = nn.Parameter(torch.zeros(1, input_dim), requires_grad=True)
        #     self.prior_net = nn.Sequential(
        #         nn.Linear(input_dim, output_dim),
        #         nn.LayerNorm(output_dim, eps=1e-3),
        #         nn.SiLU(),
        #         nn.Linear(output_dim, output_dim * 2),
        #     )
        
        if not share_backbone:
            self.backbone = StochNetBackbone(slot_dim, input_dim, 4, 4, 1024, 0.1)

        if self._bound_mean_std:
            self.max_std, self.min_std, self.init_std = 1.0, 0.1, 2.0

        if use_action:
            self.stoch_post = nn.Linear(input_dim + num_slots * slot_dim + action_dim, output_dim * 2)
            self.stoch_prior = nn.Linear(input_dim + action_dim, output_dim * 2)
        else:
            self.stoch_post = nn.Linear(input_dim + num_slots * slot_dim, output_dim * 2)
            self.stoch_prior = nn.Linear(input_dim, output_dim * 2)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._initial == "zeros":
            stoch = torch.zeros(batch_size, self.output_dim, device=device)
        elif self._initial == "gaussian":
            stoch = torch.randn(batch_size, self.output_dim, device=device) * 0.1
        # elif self._initial == "learned":
        #     deter = torch.tanh(self.W).repeat(batch_size, 1)
        #     mean, std = self.prior_net(deter).chunk(2, dim=-1)
        #     stoch = self.get_dist(mean, std).mode()
        else:
            raise ValueError(f"Invalid initial state: {self._initial}")
        return stoch.unsqueeze(1)

    def aggregate(self, slots: torch.Tensor, start: int = 0) -> torch.Tensor:
        return self.backbone(slots, start)

    def get_dist(self, mean: torch.Tensor, std: torch.Tensor) -> D.Distribution:
        return D.Independent(D.Normal(mean, std), 1, validate_args=False)

    def _forward_post(self, x: torch.Tensor, slots: torch.Tensor, actions: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        slots = rearrange(slots, "b t k d -> b t (k d)")
        slots = slots[:, 1:] - slots[:, :-1]
        x = torch.cat([x[:, :-1], slots], dim=-1)
        if self.use_action:
            actions = actions.to(x.device)
            x = torch.cat([x, actions], dim=-1)
        # x = x.detach()
        if self._bound_mean_std:
            mean_, std_ = self.stoch_post(x).chunk(2, dim=-1)
            mean = torch.tanh(mean_)
            std = (self.max_std - self.min_std) * torch.sigmoid(std_ + self.init_std) + self.min_std
        else:
            mean, log_std = self.stoch_post(x).chunk(2, dim=-1)
            std = F.softplus(log_std) + 1e-6
        return mean, std

    def _forward_prior(self, x: torch.Tensor, actions: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_action:
            actions = actions.to(x.device)
            x = torch.cat([x, actions], dim=-1)
        # x = x.detach()
        if self._bound_mean_std:
            mean_, std_ = self.stoch_prior(x).chunk(2, dim=-1)
            mean = torch.tanh(mean_)
            std = (self.max_std - self.min_std) * torch.sigmoid(std_ + self.init_std) + self.min_std
        else:
            mean, log_std = self.stoch_prior(x).chunk(2, dim=-1)
            std = F.softplus(log_std) + 1e-6
        return mean, std

    def forward(self, x: torch.Tensor, slots: Union[torch.Tensor, None] = None, actions: Union[torch.Tensor, None] = None, mode: str = "post") -> D.Distribution:
        if mode == "post":
            mean, std = self._forward_post(x, slots, actions)
        elif mode == "prior":
            mean, std = self._forward_prior(x, actions)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return self.get_dist(mean, std)


class MovingDistStochNet(nn.Module):
    def __init__(self, input_dim, num_slots, output_dim, slot_dim = None, action_dim = None, share_backbone: bool = True,
                 use_action: bool = False, initial: str = "zeros", bound_mean_std: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.output_dim = output_dim
        self.slot_dim = slot_dim
        self.action_dim = action_dim
        self.share_backbone = share_backbone
        self.use_action = use_action
        self._initial = initial
        self._bound_mean_std = bound_mean_std
        
        if self._bound_mean_std:
            self.max_std, self.min_std, self.init_std = 1.0, 0.1, 2.0

        if use_action:
            self.stoch_post = nn.Linear(input_dim + action_dim, output_dim * 2)
            self.stoch_prior = nn.Linear(input_dim + action_dim, output_dim * 2)
        else:
            self.stoch_post = nn.Linear(input_dim, output_dim * 2)
            self.stoch_prior = nn.Linear(input_dim, output_dim * 2)
            # self.stoch_post = nn.Sequential(
            #     nn.Linear(input_dim, input_dim),
            #     nn.LayerNorm(input_dim),
            #     nn.SiLU(),
            #     nn.Linear(input_dim, output_dim * 2),
            # )
            # self.stoch_prior = nn.Sequential(
            #     nn.Linear(input_dim, input_dim),
            #     nn.LayerNorm(input_dim),
            #     nn.SiLU(),
            #     nn.Linear(input_dim, output_dim * 2),
            # )

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self._initial == "zeros":
            stoch = torch.zeros(batch_size, self.output_dim, device=device)
        elif self._initial == "gaussian":
            stoch = torch.randn(batch_size, self.output_dim, device=device) * 0.1
        else:
            raise ValueError(f"Invalid initial state: {self._initial}")
        return stoch.unsqueeze(1)

    def get_dist(self, mean: torch.Tensor, std: torch.Tensor) -> D.Distribution:
        return D.Independent(D.Normal(mean, std), 1, validate_args=False)

    def _forward_post(self, x: torch.Tensor, slots: torch.Tensor, actions: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_action:
            actions = actions.to(x.device)
            x = torch.cat([x, actions], dim=-1)
        if self._bound_mean_std:
            mean_, std_ = self.stoch_post(x).chunk(2, dim=-1)
            mean_ = mean_[:, 1:] - mean_[:, :-1]
            std_ = std_[:, 1:] + std_[:, :-1]
            mean = torch.tanh(mean_)
            std = (self.max_std - self.min_std) * torch.sigmoid(std_ + self.init_std) + self.min_std
        else:
            mean, log_std = self.stoch_post(x).chunk(2, dim=-1)
            std = F.softplus(log_std) + 1e-6
        return mean, std

    def _forward_prior(self, x: torch.Tensor, actions: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_action:
            actions = actions.to(x.device)
            x = torch.cat([x, actions], dim=-1)
        if self._bound_mean_std:
            mean_, std_ = self.stoch_prior(x).chunk(2, dim=-1)
            mean = torch.tanh(mean_)
            std = (self.max_std - self.min_std) * torch.sigmoid(std_ + self.init_std) + self.min_std
        else:
            mean, log_std = self.stoch_prior(x).chunk(2, dim=-1)
            std = F.softplus(log_std) + 1e-6
        return mean, std

    def forward(self, x: torch.Tensor, slots: Union[torch.Tensor, None] = None, actions: Union[torch.Tensor, None] = None, mode: str = "post") -> D.Distribution:
        if mode == "post":
            mean, std = self._forward_post(x, slots, actions)
        elif mode == "prior":
            mean, std = self._forward_prior(x, actions)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return self.get_dist(mean, std)