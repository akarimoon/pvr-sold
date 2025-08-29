import torch
import torch.nn as nn
import torch.distributions as D
from typing import List, Tuple

from modeling.autoencoder.cnn.encoder import CnnEncoder
from modeling.autoencoder.cnn.decoder import CnnDecoder
from modeling.dreamer.utils import trunc_xavier_normal_weight_init, xavier_uniform_weight_init

class DreamerEncoder(nn.Module):
    def __init__(self,
                 num_channels: list[int],
                 kernel_sizes: list[int],
                 strides: list[int],
                 in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        if not (len(num_channels) == len(kernel_sizes) == len(strides)):
            raise ValueError(f"Expected num_channels, kernel_sizes, and strides to have the same length, but got "
                             f"{len(num_channels)}, {len(kernel_sizes)}, {len(strides)}.")

        self.encoder = CnnEncoder(num_channels, kernel_sizes, strides, in_channels, act="silu", norm=True)
        self.encoder.apply(trunc_xavier_normal_weight_init)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, T, C, H, W) -> embeddings: (B, T, F*H'*W')
        Returns convolutional feature maps per timestep.
        """
        batch_size, sequence_length, _, _, _ = images.shape
        images = images.flatten(0, 1)
        feats = self.encoder(images)
        f, h2, w2 = feats.shape[1], feats.shape[2], feats.shape[3]
        embeddings = feats.reshape(batch_size, sequence_length, f*h2*w2)
        return embeddings


class DreamerDecoder(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 image_size: Tuple[int, int],
                 num_channels: list[int],
                 kernel_sizes: list[int],
                 strides: list[int]) -> None:
        super().__init__()
        self.decoder = CnnDecoder(embedding_dim=embedding_dim, image_size=image_size,
                                  num_channels=num_channels, kernel_sizes=kernel_sizes, strides=strides, act="silu", norm=True)
        # apply trunc_xavier_normal to all layers except embeddings_to_feature_map and last ConvTranspose2d
        self.decoder.apply(trunc_xavier_normal_weight_init)
        self.decoder.embeddings_to_feature_map.apply(xavier_uniform_weight_init(1.0))
        self.decoder.decoder[-3].apply(xavier_uniform_weight_init(1.0))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: (B, T, E) -> recon: (B, T, 3, H, W)
        Returns reconstructed images for L2 loss against ground-truth.
        """
        batch_size, sequence_length, _ = embeddings.shape
        reconstructions = self.decoder(embeddings.flatten(end_dim=1)).reshape(batch_size, sequence_length, 3, *self.decoder.image_size)
        return reconstructions.clamp(0, 1)