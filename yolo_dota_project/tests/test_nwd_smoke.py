"""Smoke tests for NWD loss.

Two layers of tests:
  1. Math sanity tests using only torch (no ultralytics required).
  2. Integration tests for the patch mechanism (skipped if ultralytics not installed).

Run from project root:
    python tests/test_nwd_smoke.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Make `nwd_loss` importable without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _standalone_get_covariance_matrix(obb: torch.Tensor):
    """Local reimplementation of ultralytics' _get_covariance_matrix for tests.

    Sigma = R(theta) * diag(w^2/12, h^2/12) * R(theta)^T  is the ultralytics convention
    (uniform distribution variance), not w^2/4. We reproduce that exactly so the math
    test matches what the patched loss computes at runtime.
    """
    w, h, theta = obb[..., 2], obb[..., 3], obb[..., 4]
    cos = theta.cos()
    sin = theta.sin()
    w2 = w.pow(2) / 12.0
    h2 = h.pow(2) / 12.0
    a = w2 * cos.pow(2) + h2 * sin.pow(2)
    b = w2 * sin.pow(2) + h2 * cos.pow(2)
    c = (w2 - h2) * cos * sin
    return a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1)


def _standalone_nwd(pred, target, c_constant):
    """Pure-torch NWD using the same closed form as nwd_loss.compute_nwd_obb,
    but with our local _get_covariance_matrix so the test does not depend on
    ultralytics being installed.
    """
    EPS = 1e-7
    pred = pred.float()
    target = target.float()
    cx_a, cy_a = pred[..., :2].split(1, dim=-1)
    cx_b, cy_b = target[..., :2].split(1, dim=-1)
    a1, b1, c1 = _standalone_get_covariance_matrix(pred)
    a2, b2, c2 = _standalone_get_covariance_matrix(target)
    center_dist_sq = (cx_a - cx_b).pow(2) + (cy_a - cy_b).pow(2)
    trace_sum = a1 + b1 + a2 + b2
    trace_product = a1 * a2 + b1 * b2 + 2.0 * c1 * c2
    det1 = (a1 * b1 - c1.pow(2)).clamp_min(0.0)
    det2 = (a2 * b2 - c2.pow(2)).clamp_min(0.0)
    inner = (trace_product + 2.0 * torch.sqrt(det1 * det2 + EPS)).clamp_min(0.0)
    cross_term = 2.0 * torch.sqrt(inner + EPS)
    w2_sq = (center_dist_sq + trace_sum - cross_term).clamp_min(EPS)
    return torch.exp(-torch.sqrt(w2_sq) / c_constant)


def test_math_identity_box():
    """NWD(B, B) should be very close to 1 for any non-degenerate box."""
    cases = [
        torch.tensor([[100.0, 100.0, 30.0, 20.0, 0.0]]),
        torch.tensor([[100.0, 100.0, 30.0, 20.0, 0.5]]),
        torch.tensor([[100.0, 100.0, 5.0, 5.0, 0.0]]),     # tiny
        torch.tensor([[100.0, 100.0, 200.0, 150.0, 1.2]]), # large rotated
    ]
    for box in cases:
        v = _standalone_nwd(box, box, c_constant=64.0).item()
        assert abs(v - 1.0) < 1e-4, f"NWD(B,B) != 1 for {box.tolist()}, got {v}"
    print("[OK] identity_box: NWD(B, B) ~ 1.0 for 4 shape variants")


def test_math_far_box():
    """NWD between distant boxes should approach 0."""
    box_a = torch.tensor([[100.0, 100.0, 30.0, 20.0, 0.0]])
    box_far = torch.tensor([[800.0, 800.0, 30.0, 20.0, 0.0]])
    v = _standalone_nwd(box_a, box_far, c_constant=64.0).item()
    assert v < 0.001, f"NWD for distant boxes should be ~0, got {v}"
    print(f"[OK] far_box: NWD = {v:.6f}")


def test_math_close_box():
    """Small perturbation should give NWD in a reasonable mid-range."""
    box_a = torch.tensor([[100.0, 100.0, 30.0, 20.0, 0.0]])
    box_b = torch.tensor([[105.0, 102.0, 32.0, 21.0, 0.05]])
    v = _standalone_nwd(box_a, box_b, c_constant=64.0).item()
    assert 0.7 < v < 0.99, f"NWD for small perturb should be in (0.7, 0.99), got {v}"
    print(f"[OK] close_box: NWD = {v:.6f}")


def test_math_smoothness_vs_iou():
    """Core NWD property: for tiny objects, NWD is smoother w.r.t. pixel shift than IoU.

    With same C, NWD for a pure translation does NOT depend on object size (paper's
    scale-invariance for position errors). The advantage over IoU is smoothness, not
    severity: tiny-object IoU collapses fast (0.71 at 1px on 6x6) while NWD stays ~0.98.
    Here we verify (a) NWD is size-invariant for pure translation, and (b) the value
    is in the smooth regime, not collapsing to 0.
    """
    shift = 1.0
    tiny_a = torch.tensor([[100.0, 100.0, 6.0, 6.0, 0.0]])
    tiny_b = torch.tensor([[100.0 + shift, 100.0, 6.0, 6.0, 0.0]])
    large_a = torch.tensor([[100.0, 100.0, 100.0, 100.0, 0.0]])
    large_b = torch.tensor([[100.0 + shift, 100.0, 100.0, 100.0, 0.0]])
    nwd_tiny = _standalone_nwd(tiny_a, tiny_b, c_constant=64.0).item()
    nwd_large = _standalone_nwd(large_a, large_b, c_constant=64.0).item()
    assert abs(nwd_tiny - nwd_large) < 1e-4, \
        f"NWD should be size-invariant under pure translation: tiny={nwd_tiny}, large={nwd_large}"
    expected = math.exp(-shift / 64.0)
    assert abs(nwd_tiny - expected) < 1e-4, \
        f"NWD for 1px shift, C=64 should be exp(-1/64)≈{expected:.4f}, got {nwd_tiny}"
    assert nwd_tiny > 0.95, \
        f"NWD on tiny object with 1px shift should be smooth (>0.95), got {nwd_tiny}"
    print(f"[OK] smoothness_vs_iou: NWD={nwd_tiny:.6f} for 1px shift (size-invariant, smooth)")


def test_math_gradient_flow():
    """Loss = 1 - NWD should produce non-NaN gradients on the prediction."""
    pred = torch.tensor([[100.0, 100.0, 30.0, 20.0, 0.0]], requires_grad=True)
    target = torch.tensor([[110.0, 105.0, 35.0, 22.0, 0.1]])
    loss = (1.0 - _standalone_nwd(pred, target, c_constant=64.0)).sum()
    loss.backward()
    assert pred.grad is not None, "no gradient on pred"
    assert not torch.isnan(pred.grad).any(), f"NaN in pred.grad: {pred.grad}"
    assert pred.grad.abs().sum() > 0, "zero gradient — distance not differentiable?"
    print(f"[OK] gradient_flow: loss={loss.item():.4f}, grad={pred.grad.flatten().tolist()}")


def test_math_no_overlap_gradient():
    """Key NWD advantage over IoU: when boxes have ZERO overlap, NWD must still produce
    a meaningful (non-zero, non-saturated) gradient."""
    pred = torch.tensor([[100.0, 100.0, 10.0, 10.0, 0.0]], requires_grad=True)
    target = torch.tensor([[300.0, 300.0, 10.0, 10.0, 0.0]])  # far apart, no overlap
    loss = (1.0 - _standalone_nwd(pred, target, c_constant=64.0)).sum()
    loss.backward()
    grad_mag = pred.grad.abs().sum().item()
    assert grad_mag > 1e-6, f"NWD gradient vanished on disjoint boxes: {grad_mag}"
    print(f"[OK] no_overlap_gradient: |grad|={grad_mag:.6e} (IoU would give 0)")


def test_math_batch_consistency():
    """Batch computation should equal per-sample computation."""
    preds = torch.tensor([
        [100.0, 100.0, 30.0, 20.0, 0.0],
        [200.0, 200.0, 50.0, 40.0, 0.5],
        [50.0, 50.0, 10.0, 10.0, 1.0],
    ])
    targets = torch.tensor([
        [102.0, 99.0, 32.0, 21.0, 0.05],
        [205.0, 198.0, 52.0, 41.0, 0.55],
        [55.0, 55.0, 12.0, 11.0, 0.95],
    ])
    batch_nwd = _standalone_nwd(preds, targets, c_constant=64.0).squeeze(-1)
    for i in range(3):
        single = _standalone_nwd(preds[i:i+1], targets[i:i+1], c_constant=64.0).item()
        assert abs(batch_nwd[i].item() - single) < 1e-5, \
            f"batch[{i}]={batch_nwd[i].item()} != single={single}"
    print(f"[OK] batch_consistency: {batch_nwd.tolist()}")


def test_integration_compute_nwd_obb():
    """If ultralytics is installed, test the production function directly."""
    try:
        from nwd_loss import compute_nwd_obb
        import ultralytics  # noqa
    except ImportError as exc:
        print(f"[SKIP] integration_compute_nwd_obb: {exc}")
        return

    box = torch.tensor([[100.0, 100.0, 30.0, 20.0, 0.5]])
    box_far = torch.tensor([[500.0, 500.0, 30.0, 20.0, 0.5]])
    nwd_same = compute_nwd_obb(box, box, c_constant=64.0).item()
    nwd_far = compute_nwd_obb(box, box_far, c_constant=64.0).item()
    assert abs(nwd_same - 1.0) < 1e-4
    assert nwd_far < 0.001
    print(f"[OK] integration_compute_nwd_obb: same={nwd_same:.6f}, far={nwd_far:.6f}")


def test_integration_patch_lifecycle():
    """If ultralytics is installed, verify enable/disable hooks work."""
    try:
        from nwd_loss import enable_nwd_loss, disable_nwd_loss
        import ultralytics.utils.loss as ul_loss
    except ImportError as exc:
        print(f"[SKIP] integration_patch_lifecycle: {exc}")
        return

    original_init = ul_loss.v8OBBLoss.__init__
    enable_nwd_loss(nwd_c=50.0, nwd_weight=0.4)
    assert getattr(ul_loss.v8OBBLoss, "_nwd_patched", False)
    assert ul_loss.v8OBBLoss._nwd_c == 50.0
    enable_nwd_loss(nwd_c=80.0, nwd_weight=0.7)
    assert ul_loss.v8OBBLoss._nwd_c == 80.0
    disable_nwd_loss()
    assert not getattr(ul_loss.v8OBBLoss, "_nwd_patched", False)
    assert ul_loss.v8OBBLoss.__init__ is original_init
    print("[OK] integration_patch_lifecycle: enable / update / disable all work")


def main():
    tests = [
        test_math_identity_box,
        test_math_far_box,
        test_math_close_box,
        test_math_smoothness_vs_iou,
        test_math_gradient_flow,
        test_math_no_overlap_gradient,
        test_math_batch_consistency,
        test_integration_compute_nwd_obb,
        test_integration_patch_lifecycle,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures.append((fn.__name__, str(exc)))
            print(f"[FAIL] {fn.__name__}: {exc}")
        except Exception as exc:
            failures.append((fn.__name__, f"{type(exc).__name__}: {exc}"))
            print(f"[ERROR] {fn.__name__}: {type(exc).__name__}: {exc}")
    print()
    if failures:
        print(f"FAILED {len(failures)}/{len(tests)} tests")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print(f"PASSED {len(tests)}/{len(tests)} tests")


if __name__ == "__main__":
    main()
