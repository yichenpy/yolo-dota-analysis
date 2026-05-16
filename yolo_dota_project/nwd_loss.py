"""NWD (Normalized Wasserstein Distance) loss for YOLO11 OBB.

Integrates the closed-form Wasserstein-2 distance between rotated-bbox Gaussians
into Ultralytics' RotatedBboxLoss via monkey-patching, without modifying the
ultralytics source tree.

Reference:
    Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object
    Detection", arXiv:2110.13389, 2021. Extended for aerial imagery in
    ISPRS J P & RS, 2022.

See docs/paper/nwd.md and docs/design/nwd-integration.md for theory and design.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-7


def compute_nwd_obb(
    pred_obb: torch.Tensor,
    target_obb: torch.Tensor,
    c_constant: float = 64.0,
) -> torch.Tensor:
    """Closed-form Normalized Wasserstein Distance between two sets of OBBs.

    Each OBB is modelled as a 2D Gaussian with mean = (cx, cy) and covariance
    Sigma = R(theta) * diag(w^2/4, h^2/4) * R(theta)^T = [[a, c], [c, b]].

    Using the identity Tr(sqrt(M)) = sqrt(Tr(M) + 2*sqrt(det(M))) for SPD 2x2,
    plus cyclic trace and det multiplicativity, the Wasserstein-2 distance
    squared has the closed form:

        W2^2 = ||mu_a - mu_b||^2 + Tr(Sigma_a) + Tr(Sigma_b)
               - 2 * sqrt(Tr(Sigma_a @ Sigma_b) + 2 * sqrt(det(Sigma_a) * det(Sigma_b)))

    Then NWD = exp(-sqrt(W2^2) / C).

    Args:
        pred_obb: (N, 5) predicted OBBs in (cx, cy, w, h, angle) format.
        target_obb: (N, 5) target OBBs in (cx, cy, w, h, angle) format.
        c_constant: Normalization constant C (in pixels). Roughly the average
            object size in the dataset; for DOTA-split-lite a starting point
            is 64 px. See docs/design/nwd-integration.md section 4.4.

    Returns:
        (N, 1) tensor of NWD values in (0, 1]. 1 means perfect overlap.
        Shape matches ``probiou`` output for drop-in compatibility.
    """
    from ultralytics.utils.metrics import _get_covariance_matrix

    pred = pred_obb.float()
    target = target_obb.float()

    cx_a, cy_a = pred[..., :2].split(1, dim=-1)
    cx_b, cy_b = target[..., :2].split(1, dim=-1)

    a1, b1, c1 = _get_covariance_matrix(pred)
    a2, b2, c2 = _get_covariance_matrix(target)

    center_dist_sq = (cx_a - cx_b).pow(2) + (cy_a - cy_b).pow(2)
    trace_sum = a1 + b1 + a2 + b2

    trace_product = a1 * a2 + b1 * b2 + 2.0 * c1 * c2
    det1 = (a1 * b1 - c1.pow(2)).clamp_min(0.0)
    det2 = (a2 * b2 - c2.pow(2)).clamp_min(0.0)

    inner = (trace_product + 2.0 * torch.sqrt(det1 * det2 + EPS)).clamp_min(0.0)
    cross_term = 2.0 * torch.sqrt(inner + EPS)

    w2_sq = (center_dist_sq + trace_sum - cross_term).clamp_min(EPS)
    nwd = torch.exp(-torch.sqrt(w2_sq) / c_constant)
    return nwd


class NWDRotatedBboxLoss(torch.nn.Module):
    """RotatedBboxLoss replacement that blends ProbIoU with NWD on the IoU term.

    Mirrors ``ultralytics.utils.loss.RotatedBboxLoss.forward`` and only swaps
    the iou term from ``probiou`` to ``alpha * (1 - NWD) + (1 - alpha) * (1 - ProbIoU)``.
    DFL term is left identical to the parent implementation.
    """

    def __init__(
        self,
        reg_max: int,
        nwd_c: float = 64.0,
        nwd_weight: float = 0.5,
    ) -> None:
        super().__init__()
        from ultralytics.utils.loss import DFLoss

        if not 0.0 <= nwd_weight <= 1.0:
            raise ValueError(f"nwd_weight must be in [0, 1], got {nwd_weight}")
        if nwd_c <= 0.0:
            raise ValueError(f"nwd_c must be > 0, got {nwd_c}")

        self.reg_max = reg_max
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.nwd_c = float(nwd_c)
        self.nwd_weight = float(nwd_weight)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ):
        from ultralytics.utils.metrics import probiou
        from ultralytics.utils.tal import rbox2dist

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg = pred_bboxes[fg_mask]
        tgt_fg = target_bboxes[fg_mask]

        iou = probiou(pred_fg, tgt_fg)
        loss_probiou = ((1.0 - iou) * weight).sum() / target_scores_sum

        nwd = compute_nwd_obb(pred_fg, tgt_fg, c_constant=self.nwd_c)
        loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum

        loss_iou = self.nwd_weight * loss_nwd + (1.0 - self.nwd_weight) * loss_probiou

        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4],
                anchor_points,
                target_bboxes[..., 4:5],
                reg_max=self.dfl_loss.reg_max - 1,
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4],
                anchor_points,
                target_bboxes[..., 4:5],
            )
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


_ORIGINAL_V8OBBLOSS_INIT = None


def enable_nwd_loss(nwd_c: float = 64.0, nwd_weight: float = 0.5) -> None:
    """Patch ultralytics so v8OBBLoss uses NWDRotatedBboxLoss in place of RotatedBboxLoss.

    Idempotent: calling twice with different values updates parameters; the
    underlying patch is only applied once.

    Call this once after importing ultralytics and before ``model.train(...)``.
    The patch affects only v8OBBLoss; v8DetectionLoss and other task losses are
    untouched.
    """
    global _ORIGINAL_V8OBBLOSS_INIT
    import ultralytics.utils.loss as ul_loss

    if getattr(ul_loss.v8OBBLoss, "_nwd_patched", False):
        ul_loss.v8OBBLoss._nwd_c = float(nwd_c)
        ul_loss.v8OBBLoss._nwd_weight = float(nwd_weight)
        return

    _ORIGINAL_V8OBBLOSS_INIT = ul_loss.v8OBBLoss.__init__

    def patched_init(self, model, tal_topk: int = 10, tal_topk2=None):
        _ORIGINAL_V8OBBLOSS_INIT(self, model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.bbox_loss = NWDRotatedBboxLoss(
            reg_max=self.reg_max,
            nwd_c=ul_loss.v8OBBLoss._nwd_c,
            nwd_weight=ul_loss.v8OBBLoss._nwd_weight,
        ).to(self.device)

    ul_loss.v8OBBLoss.__init__ = patched_init
    ul_loss.v8OBBLoss._nwd_c = float(nwd_c)
    ul_loss.v8OBBLoss._nwd_weight = float(nwd_weight)
    ul_loss.v8OBBLoss._nwd_patched = True


def disable_nwd_loss() -> None:
    """Restore the original v8OBBLoss.__init__. Mostly useful for tests."""
    global _ORIGINAL_V8OBBLOSS_INIT
    import ultralytics.utils.loss as ul_loss

    if not getattr(ul_loss.v8OBBLoss, "_nwd_patched", False):
        return
    if _ORIGINAL_V8OBBLOSS_INIT is not None:
        ul_loss.v8OBBLoss.__init__ = _ORIGINAL_V8OBBLOSS_INIT
    for attr in ("_nwd_c", "_nwd_weight", "_nwd_patched"):
        if hasattr(ul_loss.v8OBBLoss, attr):
            delattr(ul_loss.v8OBBLoss, attr)
    _ORIGINAL_V8OBBLOSS_INIT = None
