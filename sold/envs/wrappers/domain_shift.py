import gym
import numpy as np
import random
import cv2

class DomainShiftWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, mode, shift_mode="hue", cifar_root="./cifar"):
        super().__init__(env)
        assert shift_mode in {"none", "hue", "background"}
        self.shift_mode = shift_mode
        self.hue_range = (-0.2, 0.2) if "train" in mode else (-0.5, 0.5)
        if shift_mode == "background":
            # self._cifar = CIFAR10(root=cifar_root, train="train" if "train" in mode else "test", download=True)
            # self._cifar_data = torch.stack(
            #     [to_tensor(img) for img, _ in self._cifar]
            # ).mul(255).byte().permute(0, 2, 3, 1).numpy()
            raise NotImplementedError
        self._delta = None
        self._bg = None

    def reset(self, **kwargs):
        self._sample_episode_shift()
        obs = self.env.reset(**kwargs)
        obs = self._transform(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._transform(obs)
        return obs, reward, done, info
    
    def _sample_episode_shift(self):
        if self.shift_mode == "hue":
            self._delta = int(
                np.clip(random.uniform(*self.hue_range) * 180, -180, 180)
            )
        elif self.shift_mode == "background":
            self._bg = random.choice(self._cifar)

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        if self.shift_mode == "hue":
            return self._shift_hue(frame)
        if self.shift_mode == "background":
            return self._replace_background(frame)
        return frame

    def _shift_hue(self, frame: np.ndarray) -> np.ndarray:
        frame = np.transpose(frame, (1, 2, 0))
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + self._delta) % 180
        return np.transpose(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), (2, 0, 1))

    def _replace_background(self, frame: np.ndarray) -> np.ndarray:
        # bg = cv2.resize(self._bg, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        # mask = np.all(frame == frame[0, 0], axis=-1)
        # frame[mask] = bg[mask]
        # return frame
        raise NotImplementedError