"""Stage ``evaluate``: segmentation against hand-drawn ground truth (TECH_SPEC 4.8).

Ground truth: labelme polygons on the orthophoto, ``<annotations_dir>/stones_gt.json``,
in pixel coordinates of the orthophoto they were drawn on (a copy is kept as
``ortho_ref.jpg`` and registered to the current orthophoto, D-019). Polygons
labelled ``stone`` are stones; a polygon labelled ``region`` bounds the evaluation. With
no ``region`` polygon, the region is the union of the stones closed by a disc of
``region_close_px`` (D-017), so mortar between drawn stones is inside but large undrawn
areas are not.

Each method's point labels are rasterised onto the same pixel grid (front-most point
per pixel). A prediction takes part if at least half of its pixels are in the region;
it is then clipped to the region. Matching is one-to-one by the Hungarian algorithm on
IoU; a pair is a true positive when IoU >= ``iou_match_threshold``.

Over-segmentation: a ground-truth stone that contains (>= ``cover_frac`` of the
prediction's area) two or more predictions. Under-segmentation: a prediction that
contains (>= ``cover_frac`` of the stone's area) two or more ground-truth stones.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.optimize import linear_sum_assignment

from stonetwin.stage import AcceptanceError, StageContext, StageError
from stonetwin.stages.segment import points_to_ortho_labels


def register_images(
    ref: np.ndarray, cur: np.ndarray, ratio: float = 0.75, ransac_px: float = 3.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Similarity mapping pixel coordinates of ``ref`` to ``cur`` (both RGB uint8).

    SIFT features, Lowe ratio test, RANSAC on a 4-DoF similarity
    (``cv2.estimateAffinePartial2D``).

    Returns:
        (2x3 matrix A with ``[x', y'] = A @ [x, y, 1]``, quality: n_matches, n_inliers,
        rms residual of inliers in ``cur`` pixels, scale).
    """
    sift = cv2.SIFT_create(nfeatures=20000)
    g = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (ref, cur)]
    (k1, d1), (k2, d2) = (sift.detectAndCompute(x, None) for x in g)
    if d1 is None or d2 is None:
        raise StageError("no features found to register the annotation image")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < ratio * n.distance]
    if len(good) < 10:
        raise StageError(f"only {len(good)} feature matches between annotation and orthophoto")
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_px, maxIters=10000
    )
    if A is None:
        raise StageError("could not register the annotation image to the orthophoto")
    inl = inl.ravel().astype(bool)
    res = (src[inl] @ A[:, :2].T + A[:, 2]) - dst[inl]
    return A, {
        "n_matches": len(good),
        "n_inliers": int(inl.sum()),
        "rms_px": float(np.sqrt((res**2).sum(1).mean())),
        "scale": float(np.hypot(A[0, 0], A[1, 0])),
    }


def map_shapes(shapes: list[dict], A: np.ndarray, pre: float, post: float) -> list[dict]:
    """Map shapes: ``p_out = post * (A @ [p_in * pre, 1])``."""
    out = []
    for sh in shapes:
        q = np.asarray(sh["points"], float) * pre
        q = (q @ A[:, :2].T + A[:, 2]) * post
        out.append({**sh, "points": q.tolist()})
    return out


def rasterize_gt(shapes: list[dict], hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray | None]:
    """Stone label raster (1..N, smaller polygon wins on overlap) and region mask or None."""
    h, w = hw
    stones = [s for s in shapes if s["label"] == "stone" and len(s["points"]) >= 3]
    region = [s for s in shapes if s["label"] == "region" and len(s["points"]) >= 3]
    polys = [np.round(np.asarray(s["points"], float)).astype(np.int32) for s in stones]
    areas = [cv2.contourArea(p.astype(np.float32)) for p in polys]
    lab = np.zeros((h, w), np.int32)
    order = np.argsort(areas)[::-1]  # large first, small overwrite
    for n in order:
        cv2.fillPoly(lab, [polys[n]], int(n) + 1)
    reg = None
    if region:
        reg = np.zeros((h, w), np.uint8)
        for s in region:
            cv2.fillPoly(reg, [np.round(np.asarray(s["points"])).astype(np.int32)], 1)
        reg = reg.astype(bool)
    return lab, reg


def derive_region(gt: np.ndarray, close_px: int) -> np.ndarray:
    """Union of ground-truth stones closed with a ``close_px`` disc.

    Holes are deliberately not filled: undrawn objects enclosed by drawn stones (a door,
    a plaque) stay outside the region.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    return cv2.morphologyEx((gt > 0).astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)


def restrict(pred: np.ndarray, region: np.ndarray, min_inside: float = 0.5) -> np.ndarray:
    """Keep predictions with >= ``min_inside`` of their area in ``region``; clip; renumber."""
    ids, tot = np.unique(pred[pred > 0], return_counts=True)
    inside = np.bincount(pred[region & (pred > 0)], minlength=pred.max() + 1)
    keep = ids[inside[ids] >= min_inside * tot]
    lut = np.zeros(pred.max() + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    out = lut[pred]
    out[~region] = 0
    return out


def match(gt: np.ndarray, pred: np.ndarray, thr: float, cover_frac: float) -> dict[str, Any]:
    """Hungarian matching on IoU between two label rasters of the same shape.

    Returns counts, precision/recall/F1 at ``thr``, mean IoU of matched pairs, over- and
    under-segmentation counts, and the list of pairs.
    """
    ng, npr = int(gt.max()), int(pred.max())
    ag = np.bincount(gt.ravel(), minlength=ng + 1).astype(float)
    ap = np.bincount(pred.ravel(), minlength=npr + 1).astype(float)
    both = (gt > 0) & (pred > 0)
    inter = np.zeros((ng + 1, npr + 1))
    np.add.at(inter, (gt[both], pred[both]), 1)
    gid = np.nonzero(ag[1:])[0] + 1  # GT stones that are present
    pid = np.nonzero(ap[1:])[0] + 1
    inter_gp = inter[np.ix_(gid, pid)]
    iou = inter_gp / (ag[gid][:, None] + ap[pid][None, :] - inter_gp)
    pairs = []
    if len(gid) and len(pid):
        r, c = linear_sum_assignment(-iou)
        pairs = [
            (int(gid[i]), int(pid[j]), float(iou[i, j]))
            for i, j in zip(r, c, strict=True)
            if iou[i, j] > 0
        ]
    tp = sum(1 for p in pairs if p[2] >= thr)
    prec = tp / len(pid) if len(pid) else 0.0
    rec = tp / len(gid) if len(gid) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    pred_in_gt = inter_gp >= cover_frac * ap[pid][None, :]  # prediction mostly inside GT stone
    gt_in_pred = inter_gp >= cover_frac * ag[gid][:, None]  # GT stone mostly inside prediction
    return {
        "n_gt": len(gid),
        "n_pred": len(pid),
        "tp": tp,
        "fp": len(pid) - tp,
        "fn": len(gid) - tp,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_iou_matched": float(np.mean([p[2] for p in pairs if p[2] >= thr])) if tp else 0.0,
        "mean_iou_assigned": float(np.mean([p[2] for p in pairs])) if pairs else 0.0,
        "over_segmented_gt": int((pred_in_gt.sum(1) >= 2).sum()),
        "under_segmenting_pred": int((gt_in_pred.sum(0) >= 2).sum()),
        "pairs": pairs,
    }


def overlay(
    img: np.ndarray, gt: np.ndarray, pred: np.ndarray, region: np.ndarray, res: dict, thr: float
) -> np.ndarray:
    """Green: GT outlines. Cyan: matched predictions. Red: unmatched predictions.
    Orange tint: missed GT stones. Outside the region is dimmed."""
    out = img.astype(float)
    out[~region] *= 0.3
    tp_gt = {g for g, _, i in res["pairs"] if i >= thr}
    tp_pr = {p for _, p, i in res["pairs"] if i >= thr}
    missed = np.isin(gt, [g for g in np.unique(gt) if g > 0 and g not in tp_gt])
    out[missed] = 0.55 * out[missed] + 0.45 * np.array([255, 140, 0])

    def edges(lab: np.ndarray, sel: np.ndarray) -> np.ndarray:
        e = np.zeros(lab.shape, bool)
        e[:-1] |= lab[:-1] != lab[1:]
        e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        return cv2.dilate((e & sel).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    pr_tp = np.isin(pred, list(tp_pr))
    out[edges(pred, (pred > 0) & ~pr_tp)] = (230, 30, 30)
    out[edges(pred, pr_tp)] = (0, 220, 255)
    out[edges(gt, gt > 0)] = (40, 230, 40)
    return np.clip(out, 0, 255).astype(np.uint8)


def run(ctx: StageContext) -> None:
    """Evaluate every segmentation method that has ``points_labels.npy``.

    Raises:
        StageError: ground truth missing.
        AcceptanceError: fewer than ``evaluate.min_gt_stones`` stones in the region.
    """
    ecfg = dict(ctx.cfg["evaluate"])
    ctx.params = ecfg
    gt_path = ctx.root / ctx.cfg["annotations_dir"] / "stones_gt.json"
    if not gt_path.is_file():
        raise StageError(f"{gt_path} missing; see data/annotations/<scene>/README.md")
    seg = ctx.stage_dir("segment")
    meta = json.loads((seg / "ortho" / "ortho_meta.json").read_text(encoding="utf-8"))
    ann = json.loads(gt_path.read_text(encoding="utf-8"))
    hw = (meta["height"], meta["width"])
    ref_path = gt_path.parent / "ortho_ref.jpg"
    reg_info: dict[str, Any] = {"method": "none (same grid)"}
    if ref_path.is_file():
        # The polygons were drawn on the orthophoto of an earlier run; without metric scale
        # every SfM run has its own gauge, so the grid moves. Register the current
        # orthophoto to a copy of the annotated one and map the polygons (D-019).
        ref = np.asarray(Image.open(ref_path).convert("RGB"))
        cur = np.asarray(Image.open(seg / "ortho" / "ortho.png").convert("RGB"))
        k = ref.shape[1] / cur.shape[1]  # register at the reference's resolution
        cur_s = cv2.resize(
            cur, (ref.shape[1], round(cur.shape[0] * k)), interpolation=cv2.INTER_AREA
        )
        A, q = register_images(ref, cur_s)
        pre = ref.shape[1] / ann["imageWidth"]
        ann["shapes"] = map_shapes(ann["shapes"], A, pre, 1.0 / k)
        reg_info = {
            "method": "SIFT + RANSAC similarity to ortho_ref.jpg",
            **q,
            "rms_px_full_res": q["rms_px"] / k,
        }
        if q["n_inliers"] < ecfg["min_registration_inliers"]:
            raise AcceptanceError(f"annotation registration has only {q['n_inliers']} inliers")
    elif (ann.get("imageHeight"), ann.get("imageWidth")) != hw:
        raise StageError(
            "annotation image size differs from the orthophoto and there is no "
            "ortho_ref.jpg to register against"
        )
    gt, region = rasterize_gt(ann["shapes"], hw)
    region_source = "polygon"
    if region is None:
        region = derive_region(gt, ecfg["region_close_px"])
        region_source = f"derived (closing {ecfg['region_close_px']} px)"
    gt = restrict(gt, region)
    ctx.inputs = {
        "ground_truth": gt_path.relative_to(ctx.root).as_posix(),
        "ortho_meta": "06_segment/ortho/ortho_meta.json",
    }

    ortho = np.asarray(Image.open(seg / "ortho" / "ortho.png").convert("RGB"))
    p_wall = np.asarray(
        o3d.io.read_point_cloud(str(ctx.stage_dir("frame") / "dense_wall.ply")).points
    )
    (ctx.out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    thr, cover = ecfg["iou_match_threshold"], ecfg["cover_frac"]
    ctx.metrics = {
        "annotation_registration": reg_info,
        "region": region_source,
        "region_area_px": int(region.sum()),
        "n_gt_in_region": int(len(np.unique(gt)) - 1),
        "iou_threshold": thr,
    }
    rows: list[list[Any]] = []
    for method in ctx.cfg["segment"]["methods"]:
        f = seg / method / "points_labels.npy"
        if not f.is_file():
            continue
        pred = restrict(points_to_ortho_labels(p_wall, np.load(f), meta), region)
        res = match(gt, pred, thr, cover)
        Image.fromarray(overlay(ortho, gt, pred, region, res, thr)).save(
            ctx.out_dir / "overlays" / f"{method}.png"
        )
        rows += [[method, g, p, round(i, 4), i >= thr] for g, p, i in res["pairs"]]
        ctx.metrics[method] = {k: v for k, v in res.items() if k != "pairs"}
        # Informational only: F1 at other IoU thresholds (the reported F1 uses `thr`).
        ious = np.array([i for _, _, i in res["pairs"]])
        sweep = {}
        for t in ecfg["iou_sweep"]:
            tp = int((ious >= t).sum())
            pr, rc = tp / max(res["n_pred"], 1), tp / max(res["n_gt"], 1)
            sweep[str(t)] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        ctx.metrics[method]["f1_by_iou_threshold"] = sweep
    with (ctx.out_dir / "matches.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "gt_id", "pred_id", "iou", "true_positive"])
        w.writerows(rows)
    if ctx.metrics["n_gt_in_region"] < ecfg["min_gt_stones"]:
        raise AcceptanceError(
            f"only {ctx.metrics['n_gt_in_region']} ground-truth stones in the region "
            f"(< {ecfg['min_gt_stones']})"
        )
