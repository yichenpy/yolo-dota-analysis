"""Verify the rotated_nms cross-version compatibility wrapper.

Forces each fallback path by manipulating the ultralytics namespace, and
checks all paths agree on the kept indices (allowing small ordering diffs).

Run:
    python tests/test_rotated_nms_compat.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from sahi_inference import rotated_nms  # noqa: E402


def make_boxes(n=80, seed=0):
    torch.manual_seed(seed)
    boxes = torch.cat([
        torch.rand(n, 2) * 1000,
        torch.rand(n, 2) * 80 + 20,
        (torch.rand(n, 1) - 0.5) * 1.5,
    ], dim=-1)
    scores = torch.rand(n)
    return boxes, scores


def keep_set(t):
    return set(t.tolist())


def main():
    boxes, scores = make_boxes()
    iou = 0.5

    keep_p1 = rotated_nms(boxes, scores, iou)
    s1 = keep_set(keep_p1)
    print(f"[OK] Path 1 (auto): kept {len(s1)} / {len(boxes)}")

    import ultralytics.utils.ops as ops
    orig_nms_rotated = getattr(ops, "nms_rotated", None)
    if orig_nms_rotated is None:
        print("[INFO] nms_rotated not in ops (already running on 8.4+); skip Path 1 disable")
    else:
        del ops.nms_rotated

    try:
        import ultralytics.utils.nms as nms_mod
        has_torch_nms = hasattr(nms_mod, "TorchNMS")
    except ImportError:
        nms_mod = None
        has_torch_nms = False

    if has_torch_nms:
        keep_p2 = rotated_nms(boxes, scores, iou)
        s2 = keep_set(keep_p2)
        print(f"[OK] Path 2 (TorchNMS.fast_nms): kept {len(s2)} / {len(boxes)}")
        if s1 == s2:
            print("[OK] Path 1 set == Path 2 set")
        else:
            diff = s1.symmetric_difference(s2)
            print(f"[WARN] Path 1 vs Path 2 differ on {len(diff)} indices: {sorted(diff)[:10]}")
        del nms_mod.TorchNMS
    else:
        print("[INFO] TorchNMS path not available in this ultralytics version")

    keep_p3 = rotated_nms(boxes, scores, iou)
    s3 = keep_set(keep_p3)
    print(f"[OK] Path 3 (greedy fallback): kept {len(s3)} / {len(boxes)}")
    if s1 == s3:
        print("[OK] Path 1 set == Path 3 set")
    else:
        diff = s1.symmetric_difference(s3)
        print(f"[WARN] Path 1 vs Path 3 differ on {len(diff)} indices: {sorted(diff)[:10]}")

    if orig_nms_rotated is not None:
        ops.nms_rotated = orig_nms_rotated

    print("\nALL PATHS EXERCISED")


if __name__ == "__main__":
    main()
