import os
os.environ['MUJOCO_GL'] = 'egl'  # Set MuJoCo rendering backend.
from envs.from_gym import make_env as make_gym_env
from envs.from_gym import make_ood_env as make_ood_gym_env
from envs.wrappers.vit import ViTWrapper
from envs.wrappers.to_tensor import ToTensor
import gym
from typing import Tuple


def make_env(suite: str, name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int,
             use_cls_token: bool, freeze_vit: bool, vit_cache_dir: str, vit_model: str = None, 
             mode: str = "train", seed: int = 0) -> gym.Env:
    if suite == 'gym':
        env = make_gym_env(name, image_size, max_episode_steps, action_repeat, seed)
    elif suite == 'mof':
        from envs.from_mof import make_env as make_mof_env
        env = make_mof_env(name, image_size, max_episode_steps, action_repeat, seed)
    elif suite == 'dmcontrol':
        from envs.from_dmcontrol import make_env as make_dmcontrol_env
        env = make_dmcontrol_env(name, image_size, max_episode_steps, action_repeat, seed)
    elif suite == 'dmcgb':
        from envs.from_dmcgb import make_env as make_dmcgb_env
        env = make_dmcgb_env(name, image_size, max_episode_steps, action_repeat, seed, mode)
    elif suite == 'survival':
        from envs.from_survival import make_env as make_survival_env
        from envs.wrappers.to_tensordict import ToTensorDict
        env = make_survival_env(name, image_size, max_episode_steps, action_repeat, seed)
        env = ToTensorDict(env)
    elif suite == 'ood_gym':
        env = make_ood_gym_env(name, image_size, max_episode_steps, action_repeat, mode, seed)
    else:
        raise ValueError(f"Unsupported environment suite: {suite}")
    env = ToTensor(env)
    if vit_model is not None:
        env = ViTWrapper(env, vit_model, use_cls_token, freeze_vit, vit_cache_dir)
    return env
