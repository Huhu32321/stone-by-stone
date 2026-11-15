"""Stage ``evaluate``: segmentation against hand-drawn ground truth (TECH_SPEC 4.8).

Ground truth: labelme polygons on the orthophoto, ``<annotations_dir>/stones_gt.json``,
in orthophoto pixel coordinates (``06_segment/ortho/ortho_meta.json``). Polygons
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


def regrid_shape(shape: dict, src: dict, dst: dict) -> dict:
    """Map a labelme shape from orthophoto grid ``src`` to grid ``dst`` (both ortho_meta).

    Pixel coordinates (x right, y down) -> wall (x, y) -> pixel coordinates on ``dst``.
    """
    pts = np.asarray(shape["points"], float)
    wx = src["x0"] + pts[:, 0] * src["gsd"]
    wy = src["y0"] - pts[:, 1] * src["gsd"]
    out = np.c_[(wx - dst["x0"]) / dst["gsd"], (dst["y0"] - wy) / dst["gsd"]]
    return {**shape, "points": out.tolist()}


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
    ann_meta_path = gt_path.parent / "ortho_meta.json"
    if ann_meta_path.is_file():
        # Polygons were drawn on the grid in ann_meta; map them through wall coordinates
        # onto the current grid, so a rerun of dense/frame does not invalidate them.
        am = json.loads(ann_meta_path.read_text(encoding="utf-8"))
        ann["shapes"] = [regrid_shape(sh, am, meta) for sh in ann["shapes"]]
    elif (ann.get("imageHeight"), ann.get("imageWidth")) != hw:
        raise StageError(
            "annotation image size differs from the orthophoto and no ortho_meta.json "
            "records the grid it was drawn on"
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
