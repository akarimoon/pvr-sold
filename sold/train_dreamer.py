import copy, os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["WANDB__SERVICE_WAIT"] = "300"
from typing import Any, Dict
import hydra
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
import gym
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

from modeling.dreamer.autoencoder import DreamerEncoder, DreamerDecoder
from modeling.dreamer.dynamics import RSSM
from modeling.dreamer.prediction import TwoHotPredictor, GaussianPredictor
from utils.instantiate import instantiate_trainer
from modeling.distributions import Moments
from utils.module import FreezeParameters
from utils.training import set_seed, print_summary, OnlineModule
from utils.visualizations import make_row, stack_rows, visualize_reward_prediction


class DreamerModule(OnlineModule):
    def __init__(self,
                 encoder: DreamerEncoder,
                 rssm: RSSM,
                 decoder: DreamerDecoder,
                 reward_predictor,
                 actor,
                 critic,
                 # learning rates and clipping
                 world_model_learning_rate: float,
                 world_model_eps: float,
                 world_model_grad_clip: float,
                 actor_learning_rate: float,
                 actor_eps: float,
                 actor_grad_clip: float,
                 critic_learning_rate: float,
                 critic_eps: float,
                 critic_grad_clip: float,
                 # imagination and returns
                 imagination_horizon: int,
                 discount_factor: float,
                 return_lambda: float,
                 # KL related
                 rep_scale: float,
                 dyn_scale: float,
                 free_nats: float,
                 # behavior entropy
                 actor_entropy_scale: float,
                 actor_gradients: str,
                 # critic target ema
                 critic_ema_decay: float,
                 # env + training loop
                 env: gym.Env,
                 max_steps: int,
                 num_seed: int,
                 update_freq: int,
                 num_updates: int,
                 eval_freq: int,
                 num_eval_episodes: int,
                 batch_size: int,
                 buffer_capacity: int,
                 save_replay_buffer: bool,
                 sequence_length: int | None = None,
                 eval_env: gym.Env = None,
                 pretrain: int = 0) -> None:

        super().__init__(env, max_steps, num_seed, update_freq, num_updates, eval_freq, num_eval_episodes, batch_size,
                         sequence_length=(sequence_length if sequence_length is not None else imagination_horizon),
                         buffer_capacity=buffer_capacity, save_replay_buffer=save_replay_buffer, eval_env=eval_env,
                         pretrain=pretrain)
        self.automatic_optimization = False

        if eval_env is not None:
            self.save_hyperparameters(logger=False, ignore=['env', 'eval_env'])
        else:
            self.save_hyperparameters(logger=False, ignore=['env'])

        self.encoder = encoder
        self.rssm = rssm(action_dim=env.action_space.shape[0])
        self.decoder = decoder(embedding_dim=self.rssm.feature_dim)
        self.reward_predictor = reward_predictor(input_dim=self.rssm.feature_dim)
        self.actor = actor(input_dim=self.rssm.feature_dim, output_dim=env.action_space.shape[0])
        self.critic = critic(input_dim=self.rssm.feature_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.world_model_learning_rate = world_model_learning_rate
        self.world_model_eps = world_model_eps
        self.world_model_grad_clip = world_model_grad_clip
        self.actor_learning_rate = actor_learning_rate
        self.actor_eps = actor_eps
        self.actor_grad_clip = actor_grad_clip
        self.critic_learning_rate = critic_learning_rate
        self.critic_eps = critic_eps
        self.critic_grad_clip = critic_grad_clip

        self.imagination_horizon = imagination_horizon
        self.discount_factor = discount_factor
        self.return_lambda = return_lambda
        self.rep_scale = rep_scale
        self.dyn_scale = dyn_scale
        self.free_nats = free_nats
        self.actor_entropy_scale = actor_entropy_scale
        self.actor_gradients = actor_gradients
        self.critic_ema_decay = critic_ema_decay

        self.return_moments = Moments()
        self.register_buffer("discounts", torch.full((1, self.imagination_horizon), self.discount_factor))
        self.discounts = torch.cumprod(self.discounts, dim=1) / self.discount_factor

        self._rssm_state = None

    def configure_optimizers(self) -> OptimizerLRScheduler:
        world_model_params = list(self.encoder.parameters()) + list(self.rssm.parameters()) \
                             + list(self.decoder.parameters()) + list(self.reward_predictor.parameters())
        return [
            torch.optim.Adam(world_model_params, lr=self.world_model_learning_rate, eps=self.world_model_eps),
            torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate, eps=self.actor_eps),
            torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate, eps=self.critic_eps),
        ]

    def training_step(self, batch, batch_index: int) -> STEP_OUTPUT:
        wm_optimizer, actor_optimizer, critic_optimizer = self.optimizers()
        images, actions, rewards = batch["obs"].squeeze(0) / 255., batch["action"].squeeze(0), batch["reward"].squeeze(0)

        # 1) World model update (encoder + rssm + decoder + reward)
        wm_optimizer.zero_grad()
        outputs, posterior = self.compute_world_model_loss(images, actions, rewards)
        self.manual_backward(outputs["world_model_loss"])  # rec + reward + kl (with balance/free-nats)
        self.clip_gradients(wm_optimizer, gradient_clip_val=self.world_model_grad_clip, gradient_clip_algorithm="norm")
        wm_optimizer.step()

        # Update the target critic network.
        for critic_param, critic_target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            critic_target_param.data.copy_((1 - self.critic_ema_decay) * critic_param.data + self.critic_ema_decay * critic_target_param.data)

        # 2) Perform latent imagination to train the actor and critic.
        lambda_returns, predicted_values_targ, predicted_values_dist, action_log_probs, action_entropies = self.imagine_ahead(posterior)

        # 3-1) Learn the actor.
        actor_optimizer.zero_grad()
        outputs |= self.compute_actor_loss(lambda_returns, predicted_values_targ, action_log_probs, action_entropies)
        self.manual_backward(outputs["actor_loss"])
        self.clip_gradients(actor_optimizer, gradient_clip_val=self.actor_grad_clip, gradient_clip_algorithm="norm")
        actor_optimizer.step()

        # 3-2) Learn the critic.
        critic_optimizer.zero_grad()
        outputs |= self.compute_critic_loss(predicted_values_dist, lambda_returns, predicted_values_targ)
        self.manual_backward(outputs["critic_loss"])
        self.clip_gradients(critic_optimizer, gradient_clip_val=self.critic_grad_clip, gradient_clip_algorithm="norm")
        critic_optimizer.step()

        # 4) Logging
        for key, value in outputs.items():
            if key.endswith("_loss"):
                self.log("train/" + key, value)
        self.log_gradients(model_names=("reward_predictor", "actor", "critic"))
        return outputs

    def compute_world_model_loss(self, images: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> Dict[str, Any]:
        """Compute reconstruction, reward, and KL losses for the world model.

        Pipeline: encoder(images)->obs_embeds; RSSM.observe -> posterior & prior; decode posterior features to recon;
        compute L2 recon loss; reward twohot NLL; balanced categorical KL with free-nats; aggregate.
        """
        self.encoder.train()
        self.rssm.train()
        self.decoder.train()
        self.reward_predictor.train()
        batch_size, sequence_length, _, _, _ = images.shape
        # Encode observations to embeddings per timestep
        embeds = self.encoder(images)  # (B, T, E)

        # RSSM observe
        posterior, prior = self.rssm.observe(embeds, actions, state=None)
        features = self.rssm.get_feat(posterior)

        # Decode features to reconstructions
        recon = self.decoder(features)
        recon_loss = F.mse_loss(recon, images)

        # Visualization: reconstruction (match SOLD style) after eval
        if self.after_eval:
            with torch.no_grad():
                true_row = make_row(images[0].detach().cpu())
                model_row = make_row(recon[0].detach().cpu())
                recon_image = stack_rows([true_row, model_row])
                self.log("reconstruction", recon_image)

        # Visualization: dynamics prediction (context=6 using posterior, then prior with GT actions)
        if self.after_eval:
            with torch.no_grad():
                num_context = 6
                # Context recon from posterior
                context_state = {"stoch": posterior["stoch"][:, :num_context], "deter": posterior["deter"][:, :num_context]}
                context_feats = self.rssm.get_feat(context_state)
                context_recon = self.decoder(context_feats)

                prev = {"stoch": posterior["stoch"][:, num_context - 1:num_context], "deter": posterior["deter"][:, num_context - 1:num_context]}
                future_states = self.rssm.imagine(actions[:, num_context:], prev)
                future_feats = self.rssm.get_feat(future_states)
                future_recon = self.decoder(future_feats)

                # Build GT vs Pred rows for context and future (match SOLD style)
                gt_context_row = make_row(images[0, :num_context].detach().cpu())
                pred_context_row = make_row(context_recon[0].detach().cpu())
                context_image = stack_rows([gt_context_row, pred_context_row])

                if sequence_length > num_context:
                    gt_future_row = make_row(images[0, num_context:].detach().cpu())
                    pred_future_row = make_row(future_recon[0].detach().cpu())
                    future_image = stack_rows([gt_future_row, pred_future_row])
                    sep = torch.ones(3, context_image.size(1), 2)
                    dyn_image = torch.cat([context_image, sep, future_image], dim=2)
                else:
                    dyn_image = context_image

                self.log("dynamics_prediction", dyn_image)

        # Reward prediction loss (TwoHot symlog)
        reward_dist = self.reward_predictor(features)
        is_first = torch.isnan(rewards)
        rew_logprob = reward_dist.log_prob(torch.nan_to_num(rewards).unsqueeze(2))
        reward_loss = -rew_logprob[~is_first].mean()

        # Visualization: reward prediction (match SOLD style)
        if self.after_eval:
            with torch.no_grad():
                reward_image = visualize_reward_prediction(
                    images[0].detach(),
                    recon[0].detach(),
                    rewards[0].detach(),
                    reward_dist.mean.squeeze(2)[0].detach(),
                )
                self.log("reward_prediction", reward_image)
        
        rep_loss = D.kl_divergence(
            self.rssm.get_dist(posterior["logit"]), self.rssm.get_dist(prior["logit"].detach())
        )
        dyn_loss = D.kl_divergence(
            self.rssm.get_dist(posterior["logit"].detach()), self.rssm.get_dist(prior["logit"])
        )
        rep_loss = torch.clamp(rep_loss, min=self.free_nats).mean()
        dyn_loss = torch.clamp(dyn_loss, min=self.free_nats).mean()
        kl_value = torch.mean(rep_loss)
        
        world_model_loss = recon_loss + reward_loss + self.rep_scale * rep_loss + self.dyn_scale * dyn_loss
        return {
            "world_model_loss": world_model_loss,
            "reconstruction_loss": recon_loss,
            "reward_loss": reward_loss,
            "rep_loss": rep_loss,
            "dyn_loss": dyn_loss,
            "kl_value": kl_value,
        }, posterior

    def imagine_ahead(self, posterior: Dict[str, torch.Tensor]) -> Any:
        """Roll out imagined trajectories from the last posterior state using current actor.

        Returns:
            lambda_returns, predicted_values_targ, predicted_values_dist, action_log_probs, action_entropies
        """
        self.encoder.eval()
        self.rssm.eval()
        self.decoder.eval()
        self.reward_predictor.eval()
        batch_size = posterior["deter"].shape[0]
        state = {k: v[:, -1:].detach() for k, v in posterior.items()}
        features_seq, actions_seq = [], []
        action_log_probs, action_entropies = [], []

        with FreezeParameters([self.reward_predictor, self.critic]):
            for t in range(self.imagination_horizon):
                # Build (B,1,E) features for policy
                prev_state = {"stoch": state["stoch"], "deter": state["deter"]}
                feats = self.rssm.get_feat(prev_state)  # (B,1,E)
                action_dist = self.actor(feats)
                action = action_dist.rsample()  # (B,1,A)
                log_prob = action_dist.log_prob(action)
                entropy = action_dist.entropy()
                action_log_probs.append(log_prob)
                action_entropies.append(entropy)
                actions_seq.append(action)

                # Transition
                state = self.rssm.img_step(state, action)
                feats = self.rssm.get_feat(state)
                features_seq.append(feats)

            imagined_features = torch.cat(features_seq, dim=1)  # (B,H,E)
            # Predicted rewards and values along imagined horizon
            predicted_rewards = self.reward_predictor(imagined_features).mean.squeeze(-1)
            predicted_values = self.critic(imagined_features).mean.squeeze(-1)

        lambda_returns = self.compute_lambda_returns(predicted_rewards, predicted_values)

        action_log_probs = torch.stack(action_log_probs, dim=1)
        action_entropies = torch.stack(action_entropies, dim=1)

        # Target and current critic distributions
        predicted_values_targ = self.critic_target(imagined_features.detach()).mean.squeeze(-1)
        predicted_values_dist = self.critic(imagined_features.detach())

        if self.after_eval:
            with torch.no_grad():
                actions = torch.stack(actions_seq, dim=1)
                self.log("train/predicted_actions_mean", actions.mean().item())
                self.log("train/predicted_actions_std", actions.std().item())
                self.log("train/predicted_rewards_mean", predicted_rewards.mean().item())
                self.log("train/predicted_rewards_std", predicted_rewards.std().item())
                self.log("train/action_entropy", action_entropies.mean().item())

                recon_imag = self.decoder(imagined_features.detach())  # (B,H,3,H,W)
                image_imag = make_row(recon_imag[0].detach().cpu())
                self.log("latent_imagination", image_imag)

        return lambda_returns, predicted_values_targ, predicted_values_dist, action_log_probs, action_entropies

    def compute_actor_loss(self, lambda_returns: torch.Tensor, predicted_values_targ: torch.Tensor,
                           action_log_probs: torch.Tensor, action_entropies: torch.Tensor) -> Dict[str, Any]:
        self.actor.train()
        self.critic.eval()
        # Compute advantage estimates (like SOLD)
        offset, invscale = self.return_moments(lambda_returns[:, :-1])
        normed_lambda_returns = (lambda_returns[:, :-1] - offset) / invscale
        normed_base = (predicted_values_targ[:, :-1] - offset) / invscale
        advantage = normed_lambda_returns - normed_base

        if self.actor_gradients == "dynamics":
            actor_return_loss = -torch.mean(self.discounts.detach()[:, :-1] * advantage)
        elif self.actor_gradients == "reinforce":
            actor_return_loss = torch.mean(action_log_probs[:, :-1] * advantage.detach())
        else:
            raise ValueError(f"Invalid actor_gradients: {self.actor_gradients}.")

        actor_entropy_loss = -torch.mean(self.discounts.detach() * action_entropies)
        return {"actor_loss": actor_return_loss + self.actor_entropy_scale * actor_entropy_loss,
                "actor_return_loss": actor_return_loss,
                "actor_entropy_loss": self.actor_entropy_scale * actor_entropy_loss}

    def compute_critic_loss(self, predicted_values_dist: D.Distribution, lambda_returns: torch.Tensor,
                            predicted_values_targ: torch.Tensor, regularization_loss_weight: float = 0.1) -> Dict[str, Any]:
        self.actor.eval()
        self.critic.train()
        return_loss = torch.mean(self.discounts.detach() * (-predicted_values_dist.log_prob(lambda_returns.detach().unsqueeze(2))))
        target_regularization_loss = torch.mean(self.discounts.detach() * (-predicted_values_dist.log_prob(predicted_values_targ.detach().unsqueeze(2))))
        return {"critic_loss": return_loss + regularization_loss_weight * target_regularization_loss,
                "critic_return_loss": return_loss,
                "critic_target_regularization_loss": regularization_loss_weight * target_regularization_loss,
                "return_mse_loss": F.mse_loss(predicted_values_dist.mean.squeeze(2), lambda_returns).item()}

    def compute_lambda_returns(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        vals = [values[:, -1:]]
        interm = rewards + self.discount_factor * values * (1 - self.return_lambda)
        for t in reversed(range(self.imagination_horizon)):
            vals.append(interm[:, t].unsqueeze(1) + self.discount_factor * self.return_lambda * vals[-1])
        ret = torch.cat(list(reversed(vals)), dim=1)[:, :-1]
        return ret

    def select_action(self, observation: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        action_dim = self.env.action_space.shape[0]
        if is_first or self._rssm_state is None:
            self._rssm_state = self.rssm.init_state(batch_size=1, device=observation.device)
            self.last_action = torch.zeros(action_dim, dtype=torch.float32, device=observation.device)
            self._feat_history = None

        if mode == "random":
            selected_action = torch.from_numpy(self.env.action_space.sample().astype(np.float32))
        else:
            # Encode current observation
            obs = (observation.unsqueeze(0).unsqueeze(1) / 255.)  # (1,1,3,H,W)
            embeds = self.encoder(obs)  # (1,1,E)
            obs_embed = embeds[:, -1:]
            
            # Filter step to update posterior belief with last action
            posterior, _ = self.rssm.obs_step(self._rssm_state, self.last_action.view(1, 1, -1).to(obs_embed.device), obs_embed)
            self._rssm_state = posterior

            # Actor over current belief features
            state = {"stoch": posterior["stoch"], "deter": posterior["deter"]}
            feats = self.rssm.get_feat(state)  # (1,1,E)
            self._feat_history = feats if self._feat_history is None else torch.cat([self._feat_history, feats], dim=1)
            action_dist = self.actor(feats)
            if mode == "train":
                selected_action = action_dist.sample().view(-1)
            elif mode == "eval":
                selected_action = action_dist.mode.view(-1)
            else:
                raise ValueError(f"Invalid mode: {mode}")

        # Clamp to bounds and store
        selected_action = selected_action.clamp_(self.env.action_space.low[0], self.env.action_space.high[0]).detach()
        self.last_action = selected_action
        return selected_action


@hydra.main(config_path="../configs", config_name="train_dreamer", version_base=None)
def train(cfg: DictConfig):
    if os.path.exists(".env"):
        from dotenv import load_dotenv
        load_dotenv(".env")

    print_summary(cfg)
    set_seed(cfg.seed)
    module = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)

    if cfg.logger.log_to_wandb and trainer.is_global_zero:
        import wandb
        wandb.init(project="dreamer", name=cfg.experiment, config=dict(cfg), sync_tensorboard=True)

    trainer.fit(module, ckpt_path=os.path.abspath(cfg.checkpoint) if cfg.checkpoint else None)

    if cfg.logger.log_to_wandb and trainer.is_global_zero:
        wandb.finish()


if __name__ == "__main__":
    train()

