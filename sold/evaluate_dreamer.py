from collections import defaultdict
import hydra
import json
from omegaconf import DictConfig
import os
from tqdm import tqdm
import torch
from torchvision.io import write_video
from torchvision import transforms
from torchvision.utils import save_image
from train_dreamer import DreamerModule
from typing import Any, Dict, List

from train_autoencoder import AutoencoderModule
from utils.training import set_seed
from utils.visualizations import make_row, stack_rows

os.environ["HYDRA_FULL_ERROR"] = "1"


@torch.no_grad()
def play_episode(dreamer: DreamerModule, mode: str = "eval") -> Dict[str, Any]:
    obs, done, info = dreamer.env.reset(), False, {}
    episode = defaultdict(list)
    episode["obs"].append(obs)
    # episode["high_res"].append(transforms.ToTensor()(sold.env.render(size=(1024, 1024)).copy()))
    episode["high_res"].append(transforms.ToTensor()(dreamer.env.render(size=(256, 256)).copy()))
    while not done:
        last_action = dreamer.select_action(obs.to(dreamer.device), is_first=len(episode["obs"]) == 1, mode=mode).cpu()
        obs, reward, done, info = dreamer.env.step(last_action)
        episode["obs"].append(obs.cpu())
        # episode["high_res"].append(transforms.ToTensor()(sold.env.render(size=(1024, 1024)).copy()))
        episode["high_res"].append(transforms.ToTensor()(dreamer.env.render(size=(256, 256)).copy()))
        episode["actions"].append(last_action)
        episode["reward"].append(reward)

    if "success" in info:
        episode["success"] = info["success"]
    return episode

def visualize_reconstruction(images, reconstructions) -> torch.Tensor:
    rows = []
    rows.append(make_row(images.cpu()))
    rows.append(make_row(reconstructions.cpu()))
    return stack_rows(rows)

@torch.no_grad()
def rollout(dreamer: DreamerModule, episode: Dict[str, Any]):
    images = torch.stack(episode["obs"]).unsqueeze(0) / 255.
    actions = torch.stack(episode["actions"]).unsqueeze(0)
    images = images.to(dreamer.device)[:, :-1]
    actions = actions.to(dreamer.device)

    num_context = 3

    embeds = dreamer.encoder(images[:, :num_context])  # (B, T, E)

    # RSSM observe
    posterior, prior = dreamer.rssm.observe(embeds, actions, state=None)
    features = dreamer.rssm.get_feat(posterior)

    # Decode features to reconstructions
    recon = dreamer.decoder(features)
    context_outputs = visualize_reconstruction(images[0, :num_context], recon[0])

    state = {k: v[:, -1:].detach() for k, v in posterior.items()}
    future_outputs = []
    for t in range(1, dreamer.imagination_horizon + 1):
        prev_state = {"stoch": state["stoch"], "deter": state["deter"]}
        state = dreamer.rssm.img_step(prev_state, actions[:, t:t+1].clone().detach())
        features = dreamer.rssm.get_feat(prev_state)
        recon = dreamer.decoder(features)
        future_output = visualize_reconstruction(images[0, num_context+t-1:num_context+t], recon[0])
        future_outputs.append(future_output)

    dynamics_image = torch.cat([context_outputs, torch.ones(3, context_outputs.size(1), 2), *future_outputs], dim=2)
    return dynamics_image
    

def get_checkpoint_files(checkpoint_path: str) -> List[str]:
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), checkpoint_path)

    if os.path.isfile(checkpoint_path):
        return [checkpoint_path] if checkpoint_path.endswith('.ckpt') else []
    elif os.path.isdir(checkpoint_path):
        return [os.path.join(checkpoint_path, file) for file in os.listdir(checkpoint_path) if file.endswith('.ckpt')]
    else:
        raise ValueError(f"The path '{checkpoint_path}' is neither a valid file nor directory.")


@hydra.main(config_path="../configs", config_name="evaluate_dreamer")
def evaluate(cfg: DictConfig):
    set_seed(cfg.seed)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_files = get_checkpoint_files(cfg.checkpoint_path)

    for i, checkpoint in enumerate(tqdm(checkpoint_files, disable=len(checkpoint_files) == 1, desc="Evaluating checkpoints")):
        env = hydra.utils.instantiate(cfg.env)
        dreamer = DreamerModule.load_from_checkpoint(checkpoint, env=env)

        # Log behavior videos.
        videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        metrics_filename = os.path.join(output_dir, "metrics.jsonl")
        episode_returns, successes = [], []
        for episode_index in range(cfg.eval_episodes):
            checkpoint_filename = os.path.splitext(os.path.basename(checkpoint))[0]
            checkpoint_videos_dir = os.path.join(videos_dir, checkpoint_filename)
            os.makedirs(checkpoint_videos_dir, exist_ok=True)
            episode = play_episode(dreamer, mode="eval")
            write_video(os.path.join(checkpoint_videos_dir, f"episode_obs_{episode_index}.mp4"),
                        (torch.stack(episode["obs"]).permute(0, 2, 3, 1)), fps=10)
            write_video(os.path.join(checkpoint_videos_dir, f"episode_high_res_{episode_index}.mp4"),
                        (torch.stack(episode["high_res"]).permute(0, 2, 3, 1) * 255).to(torch.uint8), fps=10)
            episode_returns.append(sum(episode["reward"]))

            dynamics_image = rollout(dreamer, episode)
            save_image(dynamics_image, os.path.join(checkpoint_videos_dir, f"episode_dynamics_{episode_index}.png"))

            if "success" in episode:
                successes.append(episode["success"])

        # Log return and success rate metrics.
        with open(metrics_filename, mode="a") as file:
            record = {"step": dreamer.num_steps, "checkpoint": checkpoint, "episode_returns": episode_returns,}
            if len(successes) > 0:
                record["success_rate"] = sum(successes) / len(successes)
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    evaluate()
