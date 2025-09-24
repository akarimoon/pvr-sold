import os, random, shutil, warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from termcolor import colored
from typing import Any, Dict, Optional, Union
import gym
import hydra
import matplotlib as mpl
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning import LightningModule
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT

from utils.logging import LoggingStepMixin
from utils.visualizations import make_grid
from datasets.ring_buffer import RingBufferDataset
from datasets.utils import NumUpdatesWrapper

warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")


def set_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def print_summary(cfg):
    color, attrs = "magenta", ["bold"]

    def _pprint(k, v):
        print(
			colored(f'{k+":":<15}', color, attrs=attrs), v
		)

    kvs = [
        ("Output Dir", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
		("Task", cfg.model.env.name),
		("Max steps", f"{int(cfg.model.max_steps):,}"),
		("Experiment", cfg.experiment),
	]
    for k, v in kvs:
        _pprint(k, v)


class OnlineProgressBar(RichProgressBar):
    """Custom progress bar for online RL training loops."""

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.pl_module = pl_module

    def _get_train_description(self, current_epoch: int) -> str:
        train_description = f"Steps {self.pl_module.num_steps}, Episodes {max(self.pl_module.num_episodes, 0)}"
        return train_description

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def _update(self, progress_bar_id: Optional["TaskID"], current: int, visible: bool = True) -> None:
        if self.is_enabled and self.progress is None:
            super().on_train_start(self.trainer, self.pl_module)
            self.train_progress_bar_id = self._add_task(self.total_train_batches,
                                                        self._get_train_description(self.trainer.current_epoch))
            progress_bar_id = self.train_progress_bar_id
        if self.progress is not None and self.is_enabled:
            self.progress.update(progress_bar_id, completed=self.pl_module.num_steps, visible=visible, description=self._get_train_description(self.trainer.current_epoch))
            self.refresh()

    @property
    def total_train_batches(self) -> Union[int, float]:
        return self.trainer.model.max_steps


class OnlineModule(LoggingStepMixin, LightningModule, ABC):
    def __init__(self, env: gym.Env, max_steps: int, num_seed: int, update_freq: int, num_updates: int,
                 eval_freq: int, num_eval_episodes: int, batch_size: int, sequence_length: int, buffer_capacity: int,
                 save_replay_buffer: bool, eval_env: gym.Env = None, pretrain: int = 0) -> None:
        """Integrates online experience collection with PyTorch Lightning's training loop.

        Args:
            env (gym.Env): The environment to interact with.
            max_steps (int): Maximum number of environment steps to collect before stopping training.
            num_seed (int): Number of seed steps/episodes to collect before training.
            update_freq (int): Update the agent every 'update_freq' environment steps.
            num_updates (int): Number of updates to perform whenever the agent is being updated.
            eval_freq (int): Evaluate the agent every 'eval_freq' environment steps.
            num_eval_episodes (int): Number of episodes to collect when evaluating the agent.
            batch_size (int): Batch size of experience sampled from the replay buffer.
            sequence_length (int): Length of sequences sampled from the replay buffer.
            buffer_capacity (int): Maximum number of steps to store in the replay buffer.
        """
        super().__init__()
        self.env = env
        self.max_steps = max_steps
        self.num_seed = num_seed
        self.update_freq = update_freq
        self.num_updates = num_updates
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.buffer_capacity = buffer_capacity
        self.save_replay_buffer = save_replay_buffer
        self.eval_env = eval_env
        self.pretrain_updates_remaining = pretrain

        self.last_action = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)),
                                           float('nan')).to(self.device)
        self.num_steps = 0
        self.num_episodes = -1  # Start at -1 to account for the initial reset.

    @property
    def checkpoints_dir(self) -> Path:
        checkpoints_dir = Path(self.logger.log_dir) / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    @abstractmethod
    def select_action(self, obs: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        pass

    def on_fit_start(self) -> None:
        self.eval_next = False
        self.after_eval = False
        self.obs = None
        self.done = True

    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, "replay_buffer"):
            self.replay_buffer = RingBufferDataset(self.buffer_capacity, self.batch_size, self.sequence_length,
                                                  save_path=self.logger.log_dir + "/replay_buffer")
        effective_updates = self.num_updates
        if self.pretrain_updates_remaining > 0 and self.num_steps >= self.num_seed:
            effective_updates = self.pretrain_updates_remaining
        dataset = NumUpdatesWrapper(self.replay_buffer, effective_updates)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=1)
        return dataloader

    @torch.no_grad()
    def on_train_epoch_start(self) -> None:
        num_collect_steps = self.num_seed if self.num_steps == 0 and self.num_seed > 0 else self.update_freq
        for _ in range(num_collect_steps):
            self.collect_step()

    @torch.no_grad()
    def collect_step(self) -> None:
        if self.num_steps % self.eval_freq == 0:
            self.eval_next = True

        if self.done:
            self.num_episodes += 1
            if self.eval_next:
                self.eval_loop()

            # Reset environment and store initial observation.
            self.log("train/buffer_size", self.replay_buffer.num_timesteps)
            if self.replay_buffer.last_episode_return is not None:
                self.log("train/episode_return", self.replay_buffer.last_episode_return.item(), prog_bar=True)
            self.obs = self.env.reset()
            self.replay_buffer.add_step(self._complete_first_timestep({"obs": self.obs}))
            self.log("train/num_episodes", self.num_episodes)
            self.log("train/num_steps", self.num_steps)
            if self.replay_buffer.last_episode_action_mean is not None:
                self.log("train/last_episode_action_mean", self.replay_buffer.last_episode_action_mean.mean().item())
                self.log("train/last_episode_action_std", self.replay_buffer.last_episode_action_std.mean().item())

        # Select action, perform environment step, and store resulting experience.
        mode = "train" if self.num_steps >= self.num_seed else "random"
        self.last_action[:] = self.select_action(self.obs.to(self.device), is_first=self.done, mode=mode)
        self.obs, reward, self.done, info = self.env.step(self.last_action.cpu())
        self.replay_buffer.add_step({"obs": self.obs, "action": self.last_action.cpu(), "reward": reward}, done=self.done)
        self.num_steps += 1

    def _complete_first_timestep(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        if "action" not in step_data:
            step_data["action"] = torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)),
                                                  float('nan'))
        if "reward" not in step_data:
            step_data["reward"] = torch.tensor(float('nan'))
        return step_data

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

            # Slot history heatmap: mean absolute diff over slots, per-dim over time
            if hasattr(self, "_slot_history") and self._slot_history is not None and self._slot_history.numel() > 0:
                slot_hist = self._slot_history.detach().cpu()  # [B, T, N, D]
                if slot_hist.dim() == 4 and slot_hist.size(1) > 1:
                    slot_diff = (slot_hist[:, 1:] - slot_hist[:, :-1]).abs()  # [B, T-1, N, D]
                    # per-dim over time heatmap: mean over batch and slots -> [T-1, D]
                    mean_slot_diff = slot_diff.mean(dim=0) # [T-1, N, D]
                    mean_slot_diff -= mean_slot_diff.amin(dim=2, keepdim=True)
                    mean_slot_diff /= mean_slot_diff.amax(dim=2, keepdim=True)

                    cmap = mpl.colormaps['plasma']
                    heat = torch.from_numpy(cmap(mean_slot_diff.cpu().numpy())).float()
                    heat = heat.permute(1, 3, 0, 2)[:, 0:3] # [N, 3, T-1, D]

                    heat_img = make_grid(heat, num_columns=mean_slot_diff.size(0))
                    self.log(f"eval_slot_change_heatmap_{episode_index}", heat_img)

                # per-slot mean diff (mean over time and dims)
                if 'slot_diff' in locals():
                    per_slot_mean_diff = slot_diff.mean(dim=(0, 1, 3))  # [N]
                    for i, v in enumerate(per_slot_mean_diff):
                        self.log(f"eval/slot_mean_diff_{episode_index}_{i}", float(v.item()))

            # Feature history mean diff
            if hasattr(self, "_feat_history") and self._feat_history is not None and self._feat_history.numel() > 0:
                feat_hist = self._feat_history.detach().cpu()  # [B, T, F]
                if feat_hist.dim() == 3 and feat_hist.size(1) > 1:
                    feat_diff = (feat_hist[:, 1:] - feat_hist[:, :-1]).abs()  # [B, T-1, F]
                    # per-dim over time heatmap: mean over batch and slots -> [T-1, F]
                    mean_feat_diff = feat_diff.mean(dim=0) # [T-1, F]
                    mean_feat_diff -= mean_feat_diff.amin(dim=1, keepdim=True)
                    mean_feat_diff /= mean_feat_diff.amax(dim=1, keepdim=True)

                    cmap = mpl.colormaps['plasma']
                    heat = torch.from_numpy(cmap(mean_feat_diff.cpu().numpy())).float()
                    heat = heat.unsqueeze(0).permute(0, 3, 1, 2)[:, 0:3] # [1, 3, T-1, F]

                    heat_img = make_grid(heat, num_columns=1)
                    self.log(f"eval_feat_change_heatmap_{episode_index}", heat_img)

                if 'feat_diff' in locals():
                    per_feat_mean_diff = feat_diff.mean()
                    self.log(f"eval/feat_mean_diff_{episode_index}", float(per_feat_mean_diff.item()))

        # Mean of per-timestep std across eval episodes (robust to variable lengths, no torch.nanstd)
        if len(episode_actions) > 0:
            max_T = max(t.size(0) for t in episode_actions)
            per_timestep_std_vals = []
            for t in range(max_T):
                # collect action vectors at time t from all episodes that have length > t
                vals_t = [acts[t] for acts in episode_actions if acts.size(0) > t]
                if len(vals_t) == 0:
                    continue
                vals_t = torch.stack(vals_t, dim=0)  # [N, A]
                std_t = vals_t.std(dim=0, unbiased=False)  # [A]
                per_timestep_std_vals.append(std_t.mean())
            if len(per_timestep_std_vals) > 0:
                self.log("eval/episode_actions_std", torch.stack(per_timestep_std_vals).mean())

        self.log("eval/episode_return", np.mean(episode_returns), prog_bar=True)
        if successes:
            self.log("eval/success_rate", np.mean(successes))

        for episode_index in range(3):
            episode = self.play_episode(mode="train")
            self.log(f"train/episode_{episode_index}", torch.stack(episode["obs"]))

        # Save model checkpoint.
        self.trainer.save_checkpoint(self.checkpoints_dir / f"sold-steps={self.num_steps}-episode={self.num_episodes}-"
                                                            f"eval_episode_return={np.mean(episode_returns)}.ckpt")

        self.eval_next = False
        self.after_eval = True

    @torch.no_grad()
    def play_episode(self, mode: str = "eval") -> Dict[str, Any]:
        """Run current policy for an episode to evaluate it without storing any experiences."""
        if not self.done:
            raise RuntimeError("Current training episode must have terminated before playing an episode.")

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

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.after_eval = False  # Reset 'after_eval' to False after the first training batch.

    def on_train_epoch_end(self) -> None:
        if self.num_steps >= self.max_steps:
            self.trainer.should_stop = True
        if self.pretrain_updates_remaining > 0 and self.num_steps >= self.num_seed:
            self.pretrain_updates_remaining = 0

    @property
    def logging_step(self) -> int:
        return self.num_steps

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["num_steps"] = self.num_steps
        checkpoint["num_episodes"] = self.num_episodes

        # Save the state of the replay buffer for the last checkpoint to enable checkpoint/restart (CPR) functionality.
        if self.save_replay_buffer:
            replay_buffer_dir = self.checkpoints_dir / f"replay_buffer-steps={self.num_steps}-episode={self.num_episodes}"
            if Path(self.replay_buffer.save_path).resolve().exists():
                shutil.copytree(Path(self.replay_buffer.save_path).resolve(), replay_buffer_dir)
            old_replay_buffer_dirs = sorted(list(self.checkpoints_dir.glob("replay_buffer-steps=*")))[:-1]
            for old_replay_buffer_dir in old_replay_buffer_dirs:
                shutil.rmtree(old_replay_buffer_dir)

            checkpoint["replay_buffer_path"] = replay_buffer_dir
            checkpoint["replay_buffer_capacity"] = self.replay_buffer.capacity
            checkpoint["replay_buffer_info"] = self.replay_buffer.info
            checkpoint["replay_buffer_head"] = self.replay_buffer.head
            checkpoint["replay_buffer_tail"] = self.replay_buffer.tail
            checkpoint["replay_buffer_episode_boundaries"] = self.replay_buffer.episode_boundaries

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.num_steps = checkpoint["num_steps"]
        self.num_episodes = checkpoint["num_episodes"] - 1  # Will be incremented instantly since self.done is True.

        replay_buffer_path = Path(checkpoint["replay_buffer_path"]) if "replay_buffer_path" in checkpoint else None
        if replay_buffer_path is not None and replay_buffer_path.exists():
            self.train_dataloader()  # Initializes the replay buffer.
            if checkpoint["replay_buffer_info"].fields is None:
                raise ValueError("Cannot load replay buffer without fields.")
            self.replay_buffer.load_from_files(checkpoint["replay_buffer_path"], checkpoint["replay_buffer_capacity"],
                                               checkpoint["replay_buffer_info"], checkpoint["replay_buffer_head"],
                                               checkpoint["replay_buffer_tail"],
                                               checkpoint["replay_buffer_episode_boundaries"])
