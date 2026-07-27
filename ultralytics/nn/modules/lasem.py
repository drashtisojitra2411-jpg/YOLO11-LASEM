# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight Adaptive Small-object Enhancement Module (LASEM)."""

from __future__ import annotations

import torch
from torch import nn

from .conv import Conv, DWConv

__all__ = ("LASEM",)


class LASEM(nn.Module):
    """Lightweight Adaptive Small-object Enhancement Module (LASEM).

    Enhances small-object features via multi-scale depthwise convolutions that are fused with a global-context-aware
    adaptive attention, then added back to the input through a residual connection. The module reduces channels with
    a 1x1 convolution, extracts local context in parallel with depthwise 3x3 and 5x5 convolutions, derives a global
    context vector from their combined response, and uses it to predict per-channel softmax weights that adaptively
    blend the two branches before projecting back to the original channel count.

    Attributes:
        reduce (Conv): 1x1 convolution that reduces input channels to the reduced (mid) channel count.
        dw3 (DWConv): Depthwise 3x3 convolution capturing fine-grained local context.
        dw5 (DWConv): Depthwise 5x5 convolution capturing broader local context.
        context (nn.Sequential): Global average pooling followed by a 1x1 convolution and activation that summarizes
            the fused branch response into a global context vector.
        weight (nn.Conv2d): 1x1 convolution that predicts per-branch, per-channel attention logits from the global
            context vector.
        project (Conv): 1x1 convolution that projects the fused features back to the input channel count.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import LASEM
        >>> x = torch.randn(1, 64, 40, 40)
        >>> m = LASEM(64)
        >>> y = m(x)
        >>> y.shape
        torch.Size([1, 64, 40, 40])
    """

    def __init__(self, c1: int, reduction: int = 4) -> None:
        """Initialize LASEM with given channel count and reduction ratio.

        Args:
            c1 (int): Number of input (and output) channels.
            reduction (int): Channel reduction ratio applied before the multi-scale branches.
        """
        super().__init__()
        c_mid = max(c1 // reduction, 1)
        self.reduce = Conv(c1, c_mid, k=1)
        self.dw3 = DWConv(c_mid, c_mid, k=3)
        self.dw5 = DWConv(c_mid, c_mid, k=5)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_mid, c_mid, 1, bias=True),
            nn.SiLU(),
        )
        self.weight = nn.Conv2d(c_mid, 2 * c_mid, 1, bias=True)
        self.project = Conv(c_mid, c1, k=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lightweight adaptive small-object enhancement to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, C, H, W).
        """
        identity = x
        y = self.reduce(x)
        y3 = self.dw3(y)
        y5 = self.dw5(y)

        b, c, _, _ = y.shape
        ctx = self.context(y3 + y5)
        logits = self.weight(ctx).view(b, 2, c, 1, 1)
        weights = torch.softmax(logits, dim=1)

        fused = y3 * weights[:, 0] + y5 * weights[:, 1]
        return identity + self.project(fused)
