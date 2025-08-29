from typing import Callable
import math
import torch.nn as nn


def trunc_xavier_normal_weight_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
        fan_avg = 0.5 * (fan_in + fan_out)
        std = math.sqrt(1.0 / fan_avg)
        if std is not None:
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def xavier_uniform_weight_init(given_scale: float) -> Callable[[nn.Module], None]:
    def f(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight, gain=given_scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return f