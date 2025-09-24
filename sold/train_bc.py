import os
os.environ["HYDRA_FULL_ERROR"] = "1"
from collections import defaultdict
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from omegaconf import DictConfig

from modeling.autoencoder.base import Autoencoder
from utils.instantiate import instantiate_trainer
from utils.logging import LoggingStepMixin
from scripts.open_replay_buffer import ReplayBufferReader


class BCModule(LoggingStepMixin, LightningModule):
    def __init__(self, 
                 autoencoder: Autoencoder,
                 policy,
                 env,
                 eval_env,
                 learning_rate: float,
                 weight_decay: float,
                 context: int = 3,
                 num_eval_episodes: int = 3,) -> None:
        super().__init__()

        regression_infos = {"max_episode_steps": env.max_episode_steps,  "num_slots": autoencoder.num_slots,
                            "slot_dim": autoencoder.slot_dim}

        self.env = env
        self.eval_env = eval_env
        self.autoencoder = autoencoder
        self.policy = policy(**regression_infos, output_dim=env.action_space.shape[0], lower_bound=env.action_space.low, upper_bound=env.action_space.high)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.context = context
        self.num_eval_episodes = num_eval_episodes
        self.save_hyperparameters(logger=False, ignore=["env", "eval_env"])

        for p in self.autoencoder.parameters():
            p.requires_grad = False

        # Eval episode state
        self.done = True
        self._slot_history = None
        self.last_action = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)),
                                           float('nan')).to(self.device)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params = [p for p in self.policy.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        # Expect keys from sampler collated by DataLoader: obs [B,T,H,W,3] uint8; action [B,T,A] float32
        images = batch["obs"].permute(0,1,4,2,3) / 255.0  # (B,T,3,H,W)
        actions = batch["action"].float()  # (B,T,A)

        with torch.no_grad():
            slots = self.autoencoder.encode(images, actions[:, 1:])  # (B, T, K, D)
        # Iterate over time like imagine_ahead: predict a_t from slot history up to t
        T = slots.shape[1]
        start_t = max(0, self.context - 1)
        losses = []
        for t in range(start_t, T - 1):  # predict action taken at time t, which is stored at actions[:, t+1]
            slots_ctx = slots[:, :t+1]
            dist_t = self.policy(slots_ctx, start=slots_ctx.shape[1] - 1)  # (B, 1, A) Independent
            target_t = actions[:, t+1].unsqueeze(1)  # (B, 1, A)
            log_prob_t = dist_t.log_prob(target_t)  # (B, 1) or (B,)
            losses.append(-log_prob_t.mean())
        loss = torch.stack(losses).mean()
        self.log("train/bc_loss", loss, prog_bar=True)
        return {"loss": loss}

    # TODO: add validation_step with held-out episodes and metrics (MSE and NLL)

    @torch.no_grad()
    def select_action(self, observation: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        observation = observation.unsqueeze(0) / 255.0 # (1,3,H,W)

        # Prepare prior slots and action
        last_slots = None if is_first else self._slot_history[:, -1]
        slots = self.autoencoder.encode(observation.unsqueeze(1), self.last_action.unsqueeze(0).unsqueeze(1),
                                        prior_slots=last_slots)  # Expand sequence (and batch) dimension.
        self._slot_history = slots if is_first else torch.cat([self._slot_history, slots], dim=1)

        action_dist = self.policy(self._slot_history, start=self._slot_history.shape[1] - 1)
        if mode == "train":
            selected_action = action_dist.sample().squeeze()
        else:
            selected_action = action_dist.mode.squeeze()

        return selected_action.clamp_(self.env.action_space.low[0], self.env.action_space.high[0]).detach()

    @torch.no_grad()
    def play_episode(self, mode: str = "eval") -> Dict[str, Any]:
        if mode == "eval" and self.eval_env is not None:
            env = self.eval_env
        else:
            env = self.env
        self.obs, self.done, info = env.reset(), False, {}
        episode = defaultdict(list)
        episode["obs"].append(self.obs)
        while not self.done:
            self.last_action[:] = self.select_action(self.obs.to(self.device), is_first=len(episode["obs"]) == 1,
                                                  mode=mode)
            self.obs, reward, self.done, info = env.step(self.last_action.cpu())
            episode["obs"].append(self.obs.cpu())
            episode["action"].append(self.last_action.cpu())
            episode["reward"].append(reward)

        if "success" in info:
            episode["success"] = info["success"]
        return episode

    @torch.no_grad()
    def eval_loop(self) -> None:
        episode_returns, episode_actions, successes = [], [], []
        for episode_index in range(self.num_eval_episodes):
            episode = self.play_episode(mode="eval")
            self.log(f"eval/episode_{episode_index}", torch.stack(episode["obs"]))
            episode_returns.append(sum(episode["reward"]))
            if "action" in episode and len(episode["action"]) > 0:
                episode_actions.append(torch.stack(episode["action"]))
            if "success" in episode:
                successes.append(episode["success"])

        self.log("eval/episode_return", np.mean(episode_returns), prog_bar=True)
        if successes:
            self.log("eval/success_rate", np.mean(successes))
        
        for episode_index in range(3):
            episode = self.play_episode(mode="train")
            self.log(f"train/episode_{episode_index}", torch.stack(episode["obs"]))

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        self.eval_loop()


@hydra.main(config_path="../configs", config_name="train_bc", version_base=None)
def train(cfg: DictConfig):
    # Load replay buffer from SOLD .ckpt (with metadata) or from a directory of .dat files
    if not cfg.get("ckpt_path"):
        raise ValueError("Please set ckpt_path to either a .ckpt file or a replay_buffer directory")

    ckpt_path = os.path.abspath(cfg.ckpt_path)
    image_size = tuple(getattr(cfg.env, "image_size", [64, 64])) if hasattr(cfg, "env") else (64, 64)
    reader = ReplayBufferReader.from_directory(
        directory=ckpt_path,
        capacity=None,
        image_size=(int(image_size[0]), int(image_size[1])),
    )

    class DirectoryWindowDataset(Dataset):
        def __init__(self, reader: ReplayBufferReader, sequence_length: int,
                     fraction: float = 1.0, seed: int | None = None) -> None:
            self.reader = reader
            self.T = int(sequence_length)
            any_key = next(iter(self.reader.memmaps.keys()))
            self.capacity = int(self.reader.memmaps[any_key].shape[0])
            total_windows = max(0, self.capacity - self.T + 1)
            # Build list of starting indices
            all_indices = np.arange(total_windows, dtype=np.int64)
            if fraction >= 1.0:
                self.indices = all_indices
            else:
                num = max(1, int(np.floor(total_windows * float(fraction))))
                rng = np.random.RandomState(seed if seed is not None else 0)
                self.indices = rng.choice(all_indices, size=num, replace=False)

        def __len__(self) -> int:
            return int(len(self.indices))

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            if index < 0 or index >= len(self.indices):
                raise IndexError
            start = int(self.indices[index])
            sl = slice(start, start + self.T)
            out: Dict[str, torch.Tensor] = {}
            for k, mmap in self.reader.memmaps.items():
                out[k] = torch.from_numpy(np.array(mmap[sl]))  # [T,...]
            return out

    dataset = DirectoryWindowDataset(
        reader,
        sequence_length=int(cfg.sequence_length),
        fraction=cfg.dataset_fraction,
        seed=cfg.seed,
    )
    train_loader = DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=True, pin_memory=True, num_workers=getattr(cfg, "num_workers", 1))
    # Instantiate module and trainer directly from config
    module = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)

    if cfg.logger.log_to_wandb and trainer.is_global_zero:
        import wandb
        wandb.init(project="sold_bc", name=cfg.experiment, config=dict(cfg), sync_tensorboard=True)

    trainer.fit(module, train_dataloaders=train_loader)

    if cfg.logger.log_to_wandb and trainer.is_global_zero:
        wandb.finish()

if __name__ == "__main__":
    train()

