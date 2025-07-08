import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.autoencoder.base import Autoencoder
from modeling.autoencoder.pvr4mbrl import MLPEncoder, DINOSAURDecoder
from modeling.autoencoder.savi import Corrector, SlotInitializer, Predictor
from modeling.autoencoder.savi.predictor import init_xavier_

from modeling.autoencoder.savi.predictor import init_xavier_
from utils.visualizations import make_grid, make_row, stack_rows, create_segmentation_overlay, slot_color, draw_ticks


def resize_patches_to_image(patches, size=None, scale_factor=None, resize_mode="bilinear"):
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = math.sqrt(n_patches)
    patch_size = int(math.sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = F.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])


class PVRWithSAVi(Autoencoder):
    def __init__(self, corrector: Corrector, predictor: Predictor, encoder: MLPEncoder, decoder: DINOSAURDecoder,
                 initializer: SlotInitializer) -> None:
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor
        self.encoder = encoder
        self.decoder = decoder
        self.initializer = initializer
        self._initialize_parameters()

    @torch.no_grad()
    def _initialize_parameters(self):
        init_xavier_(self)
        torch.nn.init.zeros_(self.corrector.gru.bias_ih)
        torch.nn.init.zeros_(self.corrector.gru.bias_hh)
        torch.nn.init.orthogonal_(self.corrector.gru.weight_hh)
        if hasattr(self.corrector, "slots_mu"):
            limit = math.sqrt(6.0 / (1 + self.corrector.dim_slots))
            torch.nn.init.uniform_(self.corrector.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.corrector.slots_sigma, -limit, limit)

    @property
    def num_slots(self) -> int:
        return self.corrector.num_slots

    @property
    def slot_dim(self) -> int:
        return self.corrector.slot_dim

    def encode(self, vit_features: torch.Tensor, actions: torch.Tensor, prior_slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        slots_sequence = []
        batch_size, sequence_length, _, _ = vit_features.size()

        predicted_slots = self.initializer(batch_size) if prior_slots is None else self.predictor(prior_slots, actions[:, 0])
        for time_step in range(sequence_length):
            features = self.encoder(vit_features[:, time_step])
            slots = self.corrector(features, slots=predicted_slots, is_first=prior_slots is None and time_step == 0)
            if time_step < sequence_length - 1:
                predicted_slots = self.predictor(slots, actions[:, time_step])
            slots_sequence.append(slots)

        slots_sequence = torch.stack(slots_sequence, dim=1)
        return slots_sequence

    def decode(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, num_slots, slot_dim = slots.size()
        patches, masks = self.decoder(slots.flatten(end_dim=1))
        patches = patches.view(batch_size, sequence_length, num_slots, self.decoder.num_patches, -1)

        masks_as_image = resize_patches_to_image(masks.squeeze(-1), size=self.decoder.num_patches, resize_mode="bilinear")
        masks_as_image = masks_as_image.unsqueeze(-1)
        masks_as_image = masks_as_image.view(batch_size, sequence_length, num_slots, 1, *self.decoder.image_size)
        return {"reconstructions": torch.sum(patches * masks, dim=1), "patches": patches, "masks": masks_as_image}

    def forward(self, vit_features: torch.Tensor, actions: torch.Tensor, prior_slots: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, num_patches, vit_feature_dim = vit_features.size()
        slots = self.encode(vit_features, actions, prior_slots)
        outputs = self.decode(slots)
        if "reconstructions" not in outputs:
            raise ValueError("Expected 'reconstructions' to be present in decode outputs.")
        outputs = {**outputs, "slots": slots}

        outputs["patches"] = outputs["patches"].reshape(batch_size, sequence_length, self.num_slots, num_patches, -1)
        outputs["masks"] = outputs["masks"].reshape(batch_size, sequence_length, self.num_slots, 1, *self.decoder.image_size)
        return outputs

    @torch.no_grad()
    def visualize_reconstruction(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sequence_length, num_slots, _, height, width = outputs["rgbs"].size()
        rows = []
        if "images" in outputs:
            rows.append(make_row(outputs["images"].cpu()))
        rows.append(make_row(outputs["reconstructions"].cpu()))
        images = outputs["images"].cpu() if "images" in outputs else torch.zeros_like(outputs["reconstructions"].cpu())
        rows.append(make_row(create_segmentation_overlay(images, outputs["masks"].cpu(),
                                                         background_brightness=0.0)))
        individual_slots = outputs["masks"].cpu() * outputs["rgbs"].cpu()
        rows.extend(make_row(individual_slots[:, slot_index], pad_color=slot_color(slot_index, num_slots))
                    for slot_index in range(num_slots))

        if "xticks" in outputs:
            label_backgrounds = torch.ones(sequence_length, 3, height // 3, width)
            labels = draw_ticks(label_backgrounds, outputs["xticks"], color=(0, 0, 0))
            rows.append(make_row(labels, pad_color=torch.tensor([1, 1, 1])))

        return stack_rows(rows)
