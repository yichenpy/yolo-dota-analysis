from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicChannelAttention(nn.Module):
    """Channel attention that initializes its 1x1 projection from the first input tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc: nn.Conv2d | None = None
        self.channels: int | None = None
        self.act = nn.Sigmoid()

    def _build(self, channels: int, device: torch.device) -> None:
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True).to(device=device)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = int(x.shape[1])
        if self.fc is None or self.channels != channels:
            self._build(channels, x.device)
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial attention identical to the standard CBAM spatial branch."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, 1, keepdim=True)
        max_map = torch.max(x, 1, keepdim=True)[0]
        attention = self.act(self.conv(torch.cat((avg_map, max_map), 1)))
        return x * attention


class ProjectCBAM(nn.Module):
    """Project-local CBAM that does not require channels to be hardcoded in YAML."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = DynamicChannelAttention()
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class ResidualCBFuse(nn.Module):
    """Fuse aligned shallow features into the last input with a learnable residual weight."""

    def __init__(self, alpha_init: float = 0.1) -> None:
        super().__init__()
        alpha_init = float(min(max(alpha_init, 1e-4), 1.0 - 1e-4))
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32))

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        if not xs:
            raise ValueError("ResidualCBFuse requires at least one input tensor.")
        base = xs[-1]
        residuals = []
        for tensor in xs[:-1]:
            if tensor.shape[2:] != base.shape[2:]:
                tensor = F.interpolate(tensor, size=base.shape[2:], mode="nearest")
            residuals.append(tensor)
        if not residuals:
            return base
        residual = torch.stack(residuals, dim=0).sum(dim=0)
        alpha = torch.sigmoid(self.alpha_logit).to(device=base.device, dtype=base.dtype)
        return base + alpha * residual
