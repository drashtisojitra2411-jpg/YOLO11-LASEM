# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DySample: dynamic point-sampling upsampler.

Paper: Liu, Lu, Fu, "Learning to Upsample by Learning to Sample," ICCV 2023 (arXiv:2308.15085).
Official implementation: https://github.com/tiny-smart/dysample (ported here, "lp" / point-sampling style).

Why this change: VisDrone objects frequently occupy fewer than 16x16 px. Nearest-neighbor upsampling
(YOLO11's default neck upsampler) reconstructs high-resolution feature maps without regard for object
content, which blurs the boundaries of exactly the small objects this architecture targets. DySample
predicts a content-aware sampling offset per output location instead, at a fraction of the FLOPs/params
of kernel-based dynamic upsamplers (CARAFE, FADE, SAPA), so it is used to replace every top-down
`nn.Upsample` in the Dynamic BiFPN neck (see yolo11_p2_bifpn_lasem.yaml).

Expected effect on VisDrone small-object detection: sharper, content-aligned P2-P4 feature maps after
upsampling should reduce localization error for small objects versus nearest-neighbor upsampling, at a
small constant parameter/FLOPs cost (an offset-prediction 1x1 conv) that does not scale with resolution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("DySample",)


def _normal_init(module: nn.Conv2d, std: float = 0.001) -> None:
    """Initialize a conv layer's weight from N(0, std) and its bias to zero."""
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class DySample(nn.Module):
    """Dynamic point-sampling upsampler (DySample, ICCV 2023).

    Learns a per-location sampling offset from the input feature map and reads the upsampled output via
    `grid_sample`, instead of fixed nearest-neighbor interpolation. Channel-preserving and single-input, so it
    is a drop-in replacement for `nn.Upsample` in the parser.

    Attributes:
        scale (int): Upsampling factor.
        groups (int): Number of offset groups (channels are split into this many independently-offset groups).
        enabled (bool): Ablation switch; when False, `forward` falls back to plain nearest-neighbor upsampling
            so the DySample contribution can be toggled off without editing the YAML topology.
        offset (nn.Conv2d): Predicts the (x, y) sampling offset for each of the `groups` x `scale**2` samples.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import DySample
        >>> m = DySample(64)
        >>> x = torch.randn(1, 64, 20, 20)
        >>> m(x).shape
        torch.Size([1, 64, 40, 40])
    """

    def __init__(self, c1: int, scale: int = 2, style: str = "lp", groups: int = 4, enabled: bool = True) -> None:
        """Initialize DySample.

        Args:
            c1 (int): Number of input (and output) channels.
            scale (int): Upsampling factor.
            style (str): Offset-generation style; only "lp" (point-based, the paper's lightweight default) is
                implemented since it is the variant used for detection necks.
            groups (int): Number of independently-offset channel groups; must divide `c1`.
            enabled (bool): Ablation switch, see class docstring.
        """
        super().__init__()
        if style != "lp":
            raise ValueError(f"DySample only implements style='lp', got {style!r}")
        if c1 % groups != 0:
            raise ValueError(f"DySample: c1={c1} must be divisible by groups={groups}")
        self.scale = scale
        self.groups = groups
        self.enabled = enabled

        if self.enabled:
            self.offset = nn.Conv2d(c1, 2 * groups * scale * scale, kernel_size=1)
            _normal_init(self.offset, std=0.001)
            self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self) -> torch.Tensor:
        """Build the fixed grid of initial (pre-offset) sample positions within each output pixel."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        grid = torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2).repeat(1, self.groups, 1)
        return grid.reshape(1, -1, 1, 1)

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Sample `x` at `offset`-perturbed locations and pixel-shuffle to the upsampled resolution."""
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device) + 0.5
        coords_w = torch.arange(w, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
        coords = coords.transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
            .view(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w), coords, mode="bilinear", align_corners=False, padding_mode="border"
        ).view(b, -1, self.scale * h, self.scale * w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample `x` by `scale` via learned point sampling (or nearest-neighbor when `enabled=False`).

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, C, H*scale, W*scale).
        """
        if not self.enabled:
            return F.interpolate(x, scale_factor=self.scale, mode="nearest")
        offset = self.offset(x) * 0.25 + self.init_pos
        return self._sample(x, offset)
