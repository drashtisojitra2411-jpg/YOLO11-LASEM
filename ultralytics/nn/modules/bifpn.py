# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dynamic BiFPN weighted-fusion node.

Paper: Tan, Pang, Le, "EfficientDet: Scalable and Efficient Object Detection," CVPR 2020 (arXiv:1911.09070) -
introduces BiFPN's "fast normalized fusion": each fusion input is scaled by a learnable, non-negative,
softmax-like-normalized weight (`ReLU(w_i) / (sum_j ReLU(w_j) + eps)`) before combination, so the network
learns which pyramid levels matter most instead of merging them uniformly like the plain PANet `Concat`
YOLO11 ships with. Because Ultralytics necks (and this repo's `C3k2` fusion blocks) expect
channel-concatenated inputs rather than the equal-channel elementwise sums used in the original 5-level
EfficientDet, the fusion weights are applied before `torch.cat`, matching the dominant open-source pattern
for porting BiFPN into YOLO (e.g. the widely-forked `BiFPN_Concat2`/`BiFPN_Concat3` modules for YOLOv5/v8).

The "Dynamic" extension on top of static BiFPN: each branch's static weight is additionally modulated by a
per-sample, input-content-adaptive gate (a Squeeze-and-Excitation-style gate, Hu et al., CVPR 2018) computed
from that branch's own activation statistics, so the fusion ratio can shift per image rather than being
fixed after training. This is the mechanism recent UAV small-object papers refer to as dynamic/adaptive
BiFPN weighting (e.g. IAENG IJCS 52:11 "A Small Object Detection Algorithm for UAV Aerial Images").

Why this change: VisDrone scenes mix near-uniform backgrounds (roads, fields) with dense small-object
clusters (pedestrians, vehicles); a single, image-independent fusion ratio between backbone and pyramid
features is a compromise, whereas a per-sample gate lets the network favor higher-resolution/shallower
features precisely on the images that need them, without adding a resolution-scaling amount of compute.

Expected effect on VisDrone small-object detection: better cross-scale feature fusion than static
concatenation, targeting the P2/P3 levels VisDrone small objects are detected at, at a near-negligible
parameter/FLOPs cost (two learnable vectors per node plus a scalar gate, no extra convolutions).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ("DynamicBiFPNFusion",)


class DynamicBiFPNFusion(nn.Module):
    """Multi-input dynamic BiFPN weighted-fusion node.

    Scales each of `n` same-resolution input feature maps by a learned, per-branch weight before
    concatenating them along the channel dimension. The weight is the product of a static, EfficientDet-style
    fast-normalized-fusion weight and (optionally) a per-sample content-adaptive gate, then re-normalized to
    sum to 1 across branches. Output channel count equals the sum of input channel counts (identical to a
    plain `Concat`), so it needs the same parser channel-arithmetic handling as `Concat`.

    Attributes:
        d (int): Concatenation dimension (channel axis).
        weighted (bool): Ablation switch for the static BiFPN weighting; when False, behaves as plain `Concat`.
        dynamic (bool): Ablation switch for the adaptive gate; when False, uses static weights only (vanilla
            BiFPN fast normalized fusion, no "dynamic" component).
        w (nn.Parameter): One learnable static weight per input branch.
        alpha (nn.Parameter): Per-branch gate scale (only present if `dynamic=True`).
        beta (nn.Parameter): Per-branch gate bias (only present if `dynamic=True`).

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import DynamicBiFPNFusion
        >>> m = DynamicBiFPNFusion(2)
        >>> a, b = torch.randn(1, 32, 40, 40), torch.randn(1, 64, 40, 40)
        >>> m([a, b]).shape
        torch.Size([1, 96, 40, 40])
    """

    def __init__(
        self, n: int, dimension: int = 1, weighted: bool = True, dynamic: bool = True, eps: float = 1e-4
    ) -> None:
        """Initialize the fusion node.

        Args:
            n (int): Number of input branches feeding this node.
            dimension (int): Dimension to concatenate along (1 = channel axis for NCHW tensors).
            weighted (bool): Ablation switch, see class docstring.
            dynamic (bool): Ablation switch, see class docstring.
            eps (float): Small constant preventing division by zero in weight normalization.
        """
        super().__init__()
        self.d = dimension
        self.weighted = weighted
        self.dynamic = dynamic and weighted
        self.eps = eps
        if weighted:
            self.w = nn.Parameter(torch.ones(n))
        if self.dynamic:
            self.alpha = nn.Parameter(torch.ones(n))
            self.beta = nn.Parameter(torch.zeros(n))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse `x` via weighted concatenation.

        Args:
            x (list[torch.Tensor]): Same-resolution feature maps to fuse, each of shape (B, C_i, H, W).

        Returns:
            (torch.Tensor): Concatenated tensor of shape (B, sum(C_i), H, W).
        """
        if not self.weighted:
            return torch.cat(x, self.d)

        w = torch.relu(self.w)
        if self.dynamic:
            # Per-sample, per-branch content statistic (mean activation) drives an adaptive gate on top of the
            # static weight, so the effective fusion ratio can shift per image (the "dynamic" in Dynamic BiFPN).
            ctx = torch.stack([xi.mean(dim=(1, 2, 3)) for xi in x], dim=1)  # (B, n)
            gate = torch.sigmoid(self.alpha * ctx + self.beta)  # (B, n)
            w = w.unsqueeze(0) * gate  # (B, n)
            w = w / (w.sum(dim=1, keepdim=True) + self.eps)
            return torch.cat([xi * w[:, i].view(-1, 1, 1, 1) for i, xi in enumerate(x)], self.d)

        w = w / (w.sum() + self.eps)
        return torch.cat([xi * w[i] for i, xi in enumerate(x)], self.d)
