# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Wise-IoU v3 bounding-box regression loss.

Paper: Tong, Chen, Xu, "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism," 2023
(arXiv:2301.10051).

Why this change: YOLO11's stock box loss (CIoU, see `bbox_iou` in `ultralytics/utils/metrics.py`) penalizes
every box by its own IoU/aspect-ratio/center-distance terms regardless of how "typical" that box's quality
is. VisDrone's small, densely-packed, sometimes ambiguous annotations produce many low-quality (partially
occluded/truncated) boxes; CIoU still pushes hard on these outliers, and gradient contributions from
already-easy boxes are not suppressed either. WIoU v1 replaces the aspect-ratio penalty with a distance-based
attention term whose scaling factor is explicitly detached from the graph (so it reweights, but does not
itself drive, gradients). WIoU v3 adds a non-monotonic focusing coefficient computed from each box's "outlier
degree" beta = L_IoU / running_mean(L_IoU): boxes far from the mean (in either direction - too easy or too
hard) get their loss contribution reduced, concentrating gradient on medium-quality boxes, which is where a
detector improves most.

Expected effect on VisDrone small-object detection: because small objects contribute more low-quality/
ambiguous boxes than large objects (a few-pixel localization error is a much larger relative IoU drop for a
10x10 px box than a 200x200 px one), down-weighting the outlier tail of the box-loss distribution should
stabilize small-object gradient contributions and reduce their tendency to be swamped or destabilized by a
handful of extreme-quality boxes per batch.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ("WiseIoULossV3",)


class WiseIoULossV3(nn.Module):
    """Wise-IoU v3: distance-attention IoU loss with a dynamic non-monotonic focusing mechanism.

    Maintains a running mean of the (detached) IoU loss across training batches to compute each box's outlier
    degree, per the paper's Eq. 9-11. Only meaningful during training (the running mean is only updated when
    `self.training` is True); at eval time it is used as a fixed reference.

    Attributes:
        alpha (float): Non-monotonic focusing shape parameter (paper default 1.9).
        delta (float): Non-monotonic focusing center parameter (paper default 3.0); beta == delta gives the
            maximum focusing coefficient r == 1.
        momentum (float): Exponential-moving-average momentum for the running IoU-loss mean.
        eps (float): Small constant avoiding division by zero.
        iou_mean (torch.Tensor): Running mean of the detached IoU loss (buffer, persists across batches but
            not across checkpoints by design - it re-warms quickly and need not be restored exactly).

    Examples:
        >>> import torch
        >>> from ultralytics.utils.wiou import WiseIoULossV3
        >>> loss_fn = WiseIoULossV3()
        >>> pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        >>> target = torch.tensor([[1.0, 1.0, 11.0, 11.0]])
        >>> loss_fn(pred, target).shape
        torch.Size([1, 1])
    """

    def __init__(self, alpha: float = 1.9, delta: float = 3.0, momentum: float = 1e-2, eps: float = 1e-7) -> None:
        """Initialize WiseIoULossV3 with the paper's default focusing hyperparameters.

        Args:
            alpha (float): Non-monotonic focusing shape parameter.
            delta (float): Non-monotonic focusing center parameter.
            momentum (float): EMA momentum for the running IoU-loss mean.
            eps (float): Small constant avoiding division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor) -> torch.Tensor:
        """Compute the WIoU-v3 loss between predicted and target boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted boxes, xyxy format, shape (..., 4).
            target_bboxes (torch.Tensor): Target boxes, xyxy format, shape (..., 4).

        Returns:
            (torch.Tensor): Per-box loss, shape (..., 1).
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_bboxes.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target_bboxes.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps

        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        ).clamp_(0)
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union
        l_iou = 1.0 - iou

        # R_WIoU (v1): distance-attention term, scaled by the *detached* enclosing-box diagonal so it
        # reweights L_IoU without injecting its own gradient (paper Eq. 5-6).
        with torch.no_grad():
            cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
            ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
            center_dist2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2).pow(2) + (b1_y1 + b1_y2 - b2_y1 - b2_y2).pow(2)) / 4
            r_wiou = torch.exp(center_dist2 / (cw.pow(2) + ch.pow(2)).clamp_min(self.eps))

            # Non-monotonic focusing coefficient (v3): outlier degree beta relative to a running mean of
            # L_IoU, then r = beta / (delta * alpha ** (beta - delta)) (paper Eq. 9-11).
            l_iou_detached = l_iou.detach()
            if self.training:
                self.iou_mean.mul_(1 - self.momentum).add_(self.momentum * l_iou_detached.mean())
            beta = l_iou_detached / self.iou_mean.clamp_min(self.eps)
            r = beta / (self.delta * self.alpha ** (beta - self.delta))

        return r * r_wiou * l_iou
