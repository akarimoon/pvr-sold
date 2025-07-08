import gym
import numpy as np
import torch
from einops import rearrange
from transformers import Dinov2Model


class ViTWrapper(gym.Wrapper):
    def __init__(self, env, vit_model, use_cls_token, freeze_vit, vit_cache_dir, device = torch.device("cuda")):
        super().__init__(env)
        self.vit_model = vit_model
        self.use_cls_token = use_cls_token
        self.freeze_vit = freeze_vit
        self.vit_cache_dir = vit_cache_dir
        self.device = device

        if vit_cache_dir is None:
            self.vit_encoder = Dinov2Model.from_pretrained(vit_model).to(self.device)
        else:
            self.vit_encoder = Dinov2Model.from_pretrained(vit_model, cache_dir=vit_cache_dir, local_files_only=True).to(self.device)
        
        if freeze_vit:
            for param in self.vit_encoder.parameters():
                param.requires_grad = False
        self.vit_encoder.eval()

        self.vit_outdim = self.vit_encoder.config.hidden_size
        self.h = self.observation_space.shape[-2] // self.vit_encoder.config.patch_size
        self.w = self.observation_space.shape[-1] // self.vit_encoder.config.patch_size

        if use_cls_token:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.vit_outdim,), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.h * self.w, self.vit_outdim), dtype=np.float32
            )

    def _encode(self, img):
        img = img.to(self.device).float()
        img = img / 255.0
        img -= 0.5
        if img.ndim == 3: # CHW
            img = rearrange(img, 'c h w -> 1 c h w')
        elif img.ndim == 4: # BCHW
            pass
        
        with torch.no_grad():
            if self.use_cls_token:
                # keep 2nd dimension
                img = self.vit_encoder(img).last_hidden_state[:, [0], :] # CLS token
                img = rearrange(img, 'b n d -> b (n d)')
            else:
                img = self.vit_encoder(img).last_hidden_state[:, 1:, :] # skip CLS token
        return img.squeeze(0).cpu().numpy()

    def reset(self):
        obs = self.env.reset()
        obs = self._encode(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action.to(torch.int))
        obs = self._encode(obs)
        return obs, reward, done, info