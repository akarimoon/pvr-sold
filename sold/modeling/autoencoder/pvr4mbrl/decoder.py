from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOSAURDecoder(nn.Module):
    """MLP-based Spatial Broadcast Decoder proposed by DINOSAUR (https://arxiv.org/pdf/2209.14860, Section E.1)."""
    def __init__(self, image_size: Tuple[int, int], num_patches: int, vit_feature_dim: int, hidden_dims: List[int], inp_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_patches = num_patches
        self.vit_feature_dim = vit_feature_dim
        self.slot_dim = inp_dim
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, inp_dim) * 0.02)

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(inp_dim, hidden_dim))
            layers.append(nn.ReLU())
            inp_dim = hidden_dim
        layers.append(nn.Linear(inp_dim, vit_feature_dim+1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_slots, slot_dim = slots.size()
        if slot_dim != self.slot_dim:
            raise ValueError(f"Expected slot dimension to be {self.slot_dim}, but got {slot_dim}")

        slots = slots.reshape(-1, slot_dim).unsqueeze(1).expand(-1, self.num_patches, -1)  # Spatial broadcasting -> Shape: (B * num_slots, num_patches, slot_dim)
        slots = slots + self.positional_embedding(slots)
        decoded = self.mlp(slots)
        decoded = decoded.reshape(batch_size, num_slots, self.num_patches, -1)
        patches, mask_logits = decoded.split([self.vit_feature_dim, 1], dim=2)
        masks = F.softmax(mask_logits, dim=1)
        return patches, masks
