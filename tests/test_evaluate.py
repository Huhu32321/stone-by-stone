"""Polygon rasterisation, IoU, Hungarian matching and segmentation error counts."""

from __future__ import annotations

import numpy as np
import pytest

from stonetwin.annotation import map_shapes, register_images
from stonetwin.stages.evaluate import derive_region, match, rasterize_gt, restrict


def _sq(x0: int, y0: int, x1: int, y1: int, label: str = "stone") -> dict:
    return {"label": label, "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_rasterize_and_region_polygon() -> None:
    gt, reg = rasterize_gt([_sq(10, 10, 30, 30), _sq(0, 0, 50, 50, "region")], (60, 60))
    assert gt.max() == 1 and reg is not None and reg[25, 25] and not reg[55, 55]
    assert (gt == 1).sum() == 21 * 21  # fillPoly includes both boundary rows/cols


def test_match_hand_built_case() -> None:
    gt = np.zeros((100, 100), np.int32)
    gt[0:20, 0:20] = 1  # matched well by pred 1
    gt[0:20, 40:80] = 2  # split into preds 2 and 3 -> over-segmented, unmatched at 0.5
    gt[50:70, 0:20] = 3  # merged with GT 4 into pred 4 -> under-segmentation
    gt[50:70, 22:42] = 4
    gt[80:95, 80:95] = 5  # missed entirely
    pred = np.zeros_like(gt)
    pred[0:20, 2:20] = 1  # IoU 360/400 = 0.9
    pred[0:20, 40:58] = 2  # 360 px, IoU 360/800 = 0.45
    pred[0:20, 60:80] = 3  # 400 px, IoU 0.5 exactly -> TP
    pred[50:70, 0:42] = 4  # covers GT 3 and 4
    pred[90:99, 0:9] = 5  # false positive
    r = match(gt, pred, thr=0.5, cover_frac=0.5)
    assert (r["n_gt"], r["n_pred"]) == (5, 5)
    assert r["tp"] == 2  # pred1-gt1 (0.9) and pred3-gt2 (0.5)
    assert r["precision"] == pytest.approx(2 / 5) and r["recall"] == pytest.approx(2 / 5)
    assert r["f1"] == pytest.approx(0.4)
    assert r["mean_iou_matched"] == pytest.approx((0.9 + 0.5) / 2)
    assert r["over_segmented_gt"] == 1  # GT 2 holds preds 2 and 3
    assert r["under_segmenting_pred"] == 1  # pred 4 holds GT 3 and 4
    assert {(g, p) for g, p, _ in r["pairs"]} >= {(1, 1), (2, 3)}


def test_restrict_and_derived_region() -> None:
    gt = np.zeros((60, 60), np.int32)
    gt[10:20, 10:20] = 1
    gt[10:20, 24:34] = 2  # 4 px gap to stone 1 -> closed into one region
    region = derive_region(gt, close_px=9)
    assert region[15, 22] and not region[50, 50]
    pred = np.zeros_like(gt)
    pred[10:20, 10:20] = 7  # inside
    pred[40:50, 40:50] = 9  # outside -> dropped
    out = restrict(pred, region)
    assert out.max() == 1 and (out == 1).sum() == 100


def test_register_images_recovers_similarity_and_maps_shapes() -> None:
    import cv2

    rng = np.random.default_rng(0)
    # Textured reference: random blobs, so SIFT has something to match.
    ref = np.zeros((400, 500, 3), np.uint8)
    for _ in range(300):
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.circle(
            ref,
            (int(rng.integers(0, 500)), int(rng.integers(0, 400))),
            int(rng.integers(3, 15)),
            c,
            -1,
        )
    A_true = cv2.getRotationMatrix2D((250, 200), 2.0, 1.03)
    A_true[:, 2] += (7.5, -4.0)
    cur = cv2.warpAffine(ref, A_true, (520, 420))
    A, q = register_images(ref, cur)
    assert q["n_inliers"] >= 50 and q["rms_px"] < 1.0
    pts = np.array([[100.0, 80.0], [400.0, 300.0]])
    expect = pts @ A_true[:, :2].T + A_true[:, 2]
    sh = map_shapes([{"label": "stone", "points": pts.tolist()}], A, 1.0, 1.0)
    assert np.allclose(sh[0]["points"], expect, atol=0.5)
