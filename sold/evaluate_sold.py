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
from train_sold import SOLDModule
from typing import Any, Dict, List

from train_autoencoder import AutoencoderModule
from utils.training import set_seed

os.environ["HYDRA_FULL_ERROR"] = "1"


@torch.no_grad()
def play_episode(sold: SOLDModule, mode: str = "eval") -> Dict[str, Any]:
    obs, done, info = sold.env.reset(), False, {}
    episode = defaultdict(list)
    episode["obs"].append(obs)
    # episode["high_res"].append(transforms.ToTensor()(sold.env.render(size=(1024, 1024)).copy()))
    episode["high_res"].append(transforms.ToTensor()(sold.env.render(size=(256, 256)).copy()))
    while not done:
        last_action = sold.select_action(obs.to(sold.device), is_first=len(episode["obs"]) == 1, mode=mode).cpu()
        obs, reward, done, info = sold.env.step(last_action)
        episode["obs"].append(obs.cpu())
        # episode["high_res"].append(transforms.ToTensor()(sold.env.render(size=(1024, 1024)).copy()))
        episode["high_res"].append(transforms.ToTensor()(sold.env.render(size=(256, 256)).copy()))
        episode["actions"].append(last_action)
        episode["reward"].append(reward)

    if "success" in info:
        episode["success"] = info["success"]
    return episode

@torch.no_grad()
def rollout(sold: SOLDModule, episode: Dict[str, Any]):
    images = torch.stack(episode["obs"]).unsqueeze(0) / 255.
    actions = torch.stack(episode["actions"]).unsqueeze(0)
    images = images.to(sold.device)[:, :-1]
    actions = actions.to(sold.device)

    num_context = 3
    outputs = AutoencoderModule.compute_reconstruction_loss(sold, images, actions)
    slots = outputs["slots"]
    context_slots = slots[:, :num_context].detach()
    context_outputs = sold.autoencoder.decode(context_slots)
    context_outputs["images"] = images[:, :num_context]
    context_image = sold.autoencoder.visualize_reconstruction({k: v[0] for k, v in context_outputs.items()})

    future_slots = sold.dynamics_predictor.predict_slots(context_slots, actions[:, 1:].clone().detach(), steps=sold.imagination_horizon, num_context=num_context)
    future_outputs = sold.autoencoder.decode(future_slots)
    future_outputs["images"] = images[:, num_context:num_context + sold.imagination_horizon]
    future_image = sold.autoencoder.visualize_reconstruction({k: v[0] for k, v in future_outputs.items()})
    dynamics_image = torch.cat(
        [context_image, torch.ones(3, context_image.size(1), 2), future_image], dim=2)
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


@hydra.main(config_path="../configs", config_name="evaluate_sold")
def evaluate(cfg: DictConfig):
    set_seed(cfg.seed)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    checkpoint_files = get_checkpoint_files(cfg.checkpoint_path)

    for checkpoint in tqdm(checkpoint_files, disable=len(checkpoint_files) == 1, desc="Evaluating checkpoints"):
        env = hydra.utils.instantiate(cfg.env)
        sold = SOLDModule.load_from_checkpoint(checkpoint, env=env)

        # Log behavior videos.
        videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        metrics_filename = os.path.join(output_dir, "metrics.jsonl")
        episode_returns, successes = [], []
        for episode_index in range(cfg.eval_episodes):
            checkpoint_filename = os.path.splitext(os.path.basename(checkpoint))[0]
            checkpoint_videos_dir = os.path.join(videos_dir, checkpoint_filename)
            os.makedirs(checkpoint_videos_dir, exist_ok=True)
            episode = play_episode(sold, mode="eval")
            write_video(os.path.join(checkpoint_videos_dir, f"episode_obs_{episode_index}.mp4"),
                        (torch.stack(episode["obs"]).permute(0, 2, 3, 1)), fps=10)
            write_video(os.path.join(checkpoint_videos_dir, f"episode_high_res_{episode_index}.mp4"),
                        (torch.stack(episode["high_res"]).permute(0, 2, 3, 1) * 255).to(torch.uint8), fps=10)
            episode_returns.append(sum(episode["reward"]))

            dynamics_image = rollout(sold, episode)
            save_image(dynamics_image, os.path.join(checkpoint_videos_dir, f"episode_dynamics_{episode_index}.png"))

            if "success" in episode:
                successes.append(episode["success"])

        # Log return and success rate metrics.
        with open(metrics_filename, mode="a") as file:
            record = {"step": sold.num_steps, "checkpoint": checkpoint, "episode_returns": episode_returns,}
            if len(successes) > 0:
                record["success_rate"] = sum(successes) / len(successes)
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    evaluate()
