import glob
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any, Dict, List, Tuple


class FeatureDataset(Dataset):
    def __init__(self, path: str, data_dir: str, split: str, sequence_length: int) -> None:
        super().__init__()
        self.path = path
        self.data_dir = data_dir
        self.split = split
        self.sequence_length = sequence_length
        self.split_path = os.path.join(self.path, self.data_dir, split)

        self.episode_dir_names = []
        for episode_dir in os.listdir(self.split_path):
            try:
                self.episode_dir_names.append(int(episode_dir))
            except ValueError:
                continue
        self.episode_dir_names.sort()

    @property
    def dataset_infos(self) -> Dict[str, Any]:
        if not hasattr(self, "image_size") or not hasattr(self, "action_dim"):
            image_sequence, feature_sequence, actions = self[0]
            self.image_size = list(image_sequence[0].shape[1:])
            self.action_dim = 18
        return {"image_size": self.image_size, "action_dim": self.action_dim}

    def __len__(self) -> int:
        return len(self.episode_dir_names)

    def _load_npz_data(self, index: int) -> List[np.array]:
        episode_dir = os.path.join(self.split_path, str(self.episode_dir_names[index]))
        episode_data = np.load(os.path.join(episode_dir, "episode.npz"))
        return [episode_data.get(key, None) for key in ["images", "features", "actions", "rewards"]]

class NPZDataset(FeatureDataset):
    def __init__(self, path: str, data_dir: str, split: str, sequence_length: int) -> None:
        super().__init__(path, data_dir, split, sequence_length)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        images, features, actions, rewards = self._load_npz_data(index)
        start_index = np.random.randint(0, len(features) - self.sequence_length) if self.split == "train" else 0

        image_sequence = images[start_index:start_index + self.sequence_length]
        image_sequence = (torch.from_numpy(image_sequence) / 255.).permute(0, 3, 1, 2)  # (sequence_length, 3, H, W)
        feature_sequence = features[start_index:start_index + self.sequence_length]
        feature_sequence = torch.from_numpy(feature_sequence)
        action_sequence = torch.from_numpy(actions[start_index:start_index + self.sequence_length])
        return image_sequence, feature_sequence, action_sequence


def load_feature_dataset(path: str, data_dir: str, split: str, sequence_length: int, **kwargs) -> FeatureDataset:
    return NPZDataset(path, data_dir, split, sequence_length)