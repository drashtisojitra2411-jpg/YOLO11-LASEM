# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""DySample: dynamic point-sampling upsampler.

Paper: Liu, Lu, Fu, "Learning to Upsample by Learning to Sample," ICCV 2023 (arXiv:2308.15085).
Official implementation: https://github.com/tiny-smart/dysample

`normal_init`, `DySample._init_pos`, `DySample.sample`, and `DySample.forward_lp` below are copied verbatim
from the official repository's "lp"-style path (`dyscope=False`), including variable names (`B`, `H`, `W`),
operation order, and the bare `torch.meshgrid(...)` calls with no `indexing=` kwarg. Do not "modernize" these
by adding `indexing=`, reordering permute/view/transpose calls, or renaming `sample`/`forward_lp` -- the
tensor layout flowing into `coords + offset` depends on this exact sequence matching the upstream source.
`torch.meshgrid` without `indexing=` emits a harmless `UserWarning` on newer PyTorch; it still resolves to
`indexing="ij"`, which is what the official code depends on.

The only deviations from upstream are cosmetic/interface-level, not algorithmic:
  - Constructor renamed to the project's call signature `DySample(c1, scale, style, groups, enabled)` in
    place of upstream's `DySample(in_channels, scale, style, groups, dyscope)`.
  - The `dyscope` (learned offset-scaling) option is not implemented; only plain "lp" is used by this
    project's necks, so `style` other than "lp" raises `ValueError` at construction.
  - `enabled` is a project-specific ablation switch (not in the paper/upstream): when False, `forward` falls
    back to plain nearest-neighbor upsampling so DySample can be toggled off without editing YAML topology.

Why this module exists: VisDrone objects frequently occupy fewer than 16x16 px. Nearest-neighbor upsampling
(YOLO11's default neck upsampler) reconstructs high-resolution feature maps without regard for object
content, which blurs the boundaries of exactly the small objects this architecture targets. DySample
predicts a content-aware sampling offset per output location instead, at a fraction of the FLOPs/params
of kernel-based dynamic upsamplers (CARAFE, FADE, SAPA), so it is used to replace every top-down
`nn.Upsample` in the Dynamic BiFPN neck (see yolo11_p2_bifpn_lasem.yaml).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ("DySample",)


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    """Official `normal_init`: weight ~ N(mean, std), bias set to a constant."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    """Dynamic point-sampling upsampler (DySample, ICCV 2023, "lp" style, `dyscope=False`).

    Learns a per-location sampling offset from the input feature map and reads the upsampled output via
    `grid_sample`, instead of fixed nearest-neighbor interpolation. Channel-preserving and single-input, so it
    is a drop-in replacement for `nn.Upsample` in the parser.

    Attributes:
        scale (int): Upsampling factor.
        style (str): Offset-generation style; always "lp" (enforced in `__init__`).
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
        >>> x_rect = torch.randn(1, 64, 12, 21)  # non-square input is supported
        >>> m(x_rect).shape
        torch.Size([1, 64, 24, 42])
    """

    def __init__(
        self,
        c1: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        enabled: bool = True,
    ) -> None:
        """Initialize DySample.

        Args:
            c1 (int): Number of input (and output) channels (upstream: `in_channels`).
            scale (int): Upsampling factor.
            style (str): Offset-generation style; only "lp" (point-based, the paper's lightweight default) is
                implemented since it is the variant used for detection necks.
            groups (int): Number of independently-offset channel groups; must divide `c1`.
            enabled (bool): Project-specific ablation switch, see class docstring.
        """
        super().__init__()
        if style != "lp":
            raise ValueError(f"DySample only implements style='lp', got {style!r}")
        if c1 % groups != 0:
            raise ValueError(f"DySample: c1={c1} must be divisible by groups={groups}")
        self.scale = scale
        self.style = style
        self.groups = groups
        self.enabled = enabled

        if self.enabled:
            out_channels = 2 * groups * scale**2
            self.offset = nn.Conv2d(c1, out_channels, 1)
            normal_init(self.offset, std=0.001)
            self.register_buffer("init_pos", self._init_pos(), persistent=False)

    def _init_pos(self) -> torch.Tensor:
        """Official `_init_pos`: fixed grid of initial (pre-offset) sample positions within each output pixel."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Sample x at offset-perturbed locations."""

        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5

        yy, xx = torch.meshgrid(coords_h, coords_w, indexing="ij")

        coords = torch.stack((xx, yy), dim=0)
        coords = coords.unsqueeze(1).unsqueeze(0)

        normalizer = torch.tensor(
            [W, H],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 2, 1, 1, 1)

        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale)
            .view(B, 2, -1, self.scale * H, self.scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )

        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x: torch.Tensor) -> torch.Tensor:
        """Official `forward_lp` ("lp" style, `dyscope=False` branch)."""
        offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample `x` by `scale` via learned point sampling (or nearest-neighbor when `enabled=False`).

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, C, H*scale, W*scale).
        """
        if not self.enabled:
            return F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return self.forward_lp(x)
