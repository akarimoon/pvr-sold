from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions.kl import register_kl, kl_divergence

from modeling.dreamer.utils import trunc_xavier_normal_weight_init, xavier_uniform_weight_init

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, norm: bool = True, update_bias: float = -1.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.update_bias = update_bias
        layers = []
        layers.append(nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=False))
        if norm:
            layers.append(nn.LayerNorm(3 * hidden_size, eps=1e-3))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        parts = self.layers(torch.cat([x, h], -1))
        reset, cand, update = torch.split(parts, [self.hidden_size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update + self.update_bias)
        h_new = update * cand + (1.0 - update) * h
        return h_new


class OneHotUnimixCategorical(D.OneHotCategorical):
    def __init__(self, logits: torch.Tensor = None, probs: torch.Tensor = None, unimix: float = 0.01):
        if logits is not None:
            probs = torch.softmax(logits, dim=-1)
        if probs is None:
            raise ValueError("Either logits or probs must be provided.")
        probs = (1.0 - unimix) * probs + unimix / probs.shape[-1]
        self._unimix = unimix
        super().__init__(probs=probs)

    def mode(self) -> torch.Tensor:
        _mode = F.one_hot(torch.argmax(super().logits, dim=-1), num_classes=self.logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=torch.Size()):
        y = super().sample(sample_shape)
        # Straight-through: replace gradient with probs
        return (y - self.probs).detach() + self.probs


@register_kl(OneHotUnimixCategorical, OneHotUnimixCategorical)
def _kl_onehot_unimix_onehot_unimix(p: OneHotUnimixCategorical, q: OneHotUnimixCategorical):
    p_cat = D.Categorical(probs=p.probs)
    q_cat = D.Categorical(probs=q.probs)
    return kl_divergence(p_cat, q_cat)


class RSSM(nn.Module):
    def __init__(self,
                 action_dim: int,
                 stoch_dim: int = 32,
                 deter_dim: int = 200,
                 hidden_dim: int = 200,
                 obs_embed_dim: int = 1024,
                 discrete: int = 32,
                 unimix: float = 0.01,
                 initial: str = "zeros") -> None:
        """Recurrent State-Space Model with discrete stochastic state (DreamerV3-style)."""
        super().__init__()
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.obs_embed_dim = obs_embed_dim
        self.discrete = discrete
        self.unimix = unimix
        self._initial = initial

        # Posterior MLP over concat([deter, obs_embed]) q(s_t | deter_t, obs_t)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + obs_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * discrete),
        )

        # Recurrent deterministic transition
        self.img_in = nn.Sequential(
            nn.Linear(stoch_dim * discrete + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU(),
        )
        self.gru = GRUCell(input_size=hidden_dim, hidden_size=deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-3),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * discrete),
        )

        if self._initial == "learned":
            self.W = nn.Parameter(torch.zeros(1, deter_dim), requires_grad=True)

        self.post_net.apply(trunc_xavier_normal_weight_init)
        self.img_in.apply(trunc_xavier_normal_weight_init)
        self.gru.apply(trunc_xavier_normal_weight_init)
        self.prior_net.apply(trunc_xavier_normal_weight_init)
        self.post_net[-1].apply(xavier_uniform_weight_init(1.0))
        self.prior_net[-1].apply(xavier_uniform_weight_init(1.0))

    @property
    def feature_dim(self) -> int:
        # get_feat returns concat([flattened one-hot stoch, deter])
        return self.deter_dim + self.stoch_dim * self.discrete

    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        logit = torch.zeros(batch_size, self.stoch_dim, self.discrete, device=device)

        if self._initial == "zeros":
            deter = torch.zeros(batch_size, self.deter_dim, device=device)
            stoch = torch.zeros(batch_size, self.stoch_dim, self.discrete, device=device)
        elif self._initial == "learned":
            deter = torch.tanh(self.W).repeat(batch_size, 1)
            logits = self.prior_net(deter).view(-1, self.stoch_dim, self.discrete)
            stoch = self.get_dist(logits).mode()
        else:
            raise ValueError(f"Invalid initial state: {self._initial}")
        return {"deter": deter.unsqueeze(1), "logit": logit.unsqueeze(1), "stoch": stoch.unsqueeze(1)}

    def get_feat(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get features from stacked state dict.
        state: dict with keys deter (B, T, dim), stoch (B, T, dim, discrete)
        Returns (B, T, dim + dim * discrete)
        """
        stoch = state["stoch"].flatten(start_dim=2)
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, logits: torch.Tensor) -> OneHotUnimixCategorical:
        return OneHotUnimixCategorical(logits=logits, unimix=self.unimix)

    def obs_step(self, prev: Dict[str, torch.Tensor] | None, action: torch.Tensor, obs_embed: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """One posterior step conditioned on observation.
        prev: dict with keys deter, stoch
        action: (B, 1, A)
        obs_embed: (B, 1, obs_embed_dim)
        Returns posterior state and prior state (both with mean,std,stoch,deter).
        """
        batch_size = obs_embed.shape[0]
        device = obs_embed.device

        # Initialize previous state if None
        if prev is None:
            prev = self.init_state(batch_size, device)

        # Handle sequences starting at t=0 within a batch: detect NaN actions and reset per-sample
        is_first = torch.isnan(action).any(dim=-1)
        if is_first.any():
            init = self.init_state(batch_size, device)
            prev = {
                k: torch.where(is_first.view(-1, *([1] * (v.dim() - 1))), init[k], v)
                for k, v in prev.items()
            }
            action = action.clone()
            action[is_first] = 0.0

        # Deterministic update via prior step
        prior = self.img_step(prev, action)

        # Posterior
        post_logits = self.post_net(torch.cat([prior["deter"], obs_embed], -1)).view(batch_size, -1, self.stoch_dim, self.discrete)
        stoch = self.get_dist(post_logits).sample()
        posterior = {"deter": prior["deter"], "logit": post_logits, "stoch": stoch}
        return posterior, prior

    def img_step(self, prev: Dict[str, torch.Tensor], action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One prior (imagination) step without observation."""
        batch_size = action.shape[0]
        prev_stoch = prev["stoch"].flatten(start_dim=2)
        x = torch.cat([prev_stoch, action], -1)
        x = self.img_in(x)
        deter = self.gru(x, prev["deter"])
        logits = self.prior_net(deter).view(batch_size, -1, self.stoch_dim, self.discrete)
        stoch = self.get_dist(logits).sample()
        return {"deter": deter, "logit": logits, "stoch": stoch}

    def observe(self, obs_embeds: torch.Tensor, actions: torch.Tensor, state: Dict[str, torch.Tensor] | None = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Filter a sequence of observations.
        obs_embeds: (B, T, E_obs)
        actions: (B, T, A)
        init: state dict at t=0, if None will be initialized on first obs_step
        Returns posteriors and priors as dict of tensors shaped (B, T, dim)
        """
        _, sequence_length, _ = obs_embeds.shape
        posteriors = []
        priors = []
        for i in range(sequence_length):
            posterior, prior = self.obs_step(state, actions[:, i:i+1], obs_embeds[:, i:i+1])
            posteriors.append({k: v.squeeze(1) for k, v in posterior.items()})
            priors.append({k: v.squeeze(1) for k, v in prior.items()})
            # For recurrent state, keep the (B,1,...) time dimension for the next step
            state = posterior
        return self._stack_seq(posteriors), self._stack_seq(priors)

    def imagine(self, actions: torch.Tensor, init: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Rollout prior given actions.
        actions: (B, T, A)
        init: state dict
        Returns priors as dict of tensors shaped (B, T, dim)
        """
        _, sequence_length, _ = actions.shape
        priors = []
        state = init
        for i in range(sequence_length):
            state = self.img_step(state, actions[:, i:i+1])
            priors.append({k: v.squeeze(1) for k, v in state.items()})
        return self._stack_seq(priors)

    def _stack_seq(self, states_list: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = states_list[0].keys()
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.stack([s[k] for s in states_list], dim=1)
        return out



