"""Stage ``segment``: stone instances on ``dense_wall.ply``.

Both methods write ``<method>/points_labels.npy``: one int32 per point of
``04_frame/dense_wall.ply``, 0 = mortar or unassigned, 1..N = stone ID.

Method ``ortho`` (baseline): orthographic projection of the wall-frame cloud along -z
into an RGB orthophoto and a relief map, SAM 2 automatic masks on the orthophoto,
filtering, overlap resolution, and back-projection to points through the point-to-pixel
index.

Orthophoto pixel geometry (``ortho_meta.json``): pixel (row r, col c) covers wall
coordinates ``x in [x0 + c*gsd, x0 + (c+1)*gsd)``, ``y in (y0 - (r+1)*gsd, y0 - r*gsd]``;
``x0`` is the left edge, ``y0`` the top edge. Units are scene units (relative; D-020).

Method ``multiview``: see :mod:`stonetwin.stages.segment_multiview`.

Exclusions (D-021): polygons labelled ``exclude`` in ``<annotations_dir>/<exclude_file>``
mark non-stone objects (plaques, a door). They are drawn on the annotated orthophoto and
mapped to this run's grid like the ground truth (D-019). For both methods, a stone with
at least ``exclude_min_inside`` of its pixels (ortho) or points (multiview) inside an
exclusion is removed; the rest are cut back to outside the exclusions.
"""

from __future__ import annotations

import json
import time
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from stonetwin.stage import AcceptanceError, StageContext, StageError


# --------------------------------------------------------------------------- raster
def rasterize(
    p_wall: np.ndarray, rgb: np.ndarray, gsd: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Orthographic projection along -z, keeping the front-most point per pixel.

    Args:
        p_wall: points in the wall frame (N, 3).
        rgb: colours in [0, 1] (N, 3).
        gsd: pixel size, in the units of ``p_wall``.
    Returns:
        (image uint8 HxWx3, relief float HxW with NaN where empty,
         pixel index int (N, 2) as (row, col), meta dict).
    """
    x0 = float(p_wall[:, 0].min())
    y0 = float(p_wall[:, 1].max())
    col = np.floor((p_wall[:, 0] - x0) / gsd).astype(np.int64)
    row = np.floor((y0 - p_wall[:, 1]) / gsd).astype(np.int64)
    W, H = int(col.max()) + 1, int(row.max()) + 1
    img = np.zeros((H, W, 3), np.uint8)
    relief = np.full((H, W), np.nan)
    order = np.argsort(p_wall[:, 2])  # back to front; later writes win
    img[row[order], col[order]] = np.clip(rgb[order] * 255 + 0.5, 0, 255).astype(np.uint8)
    relief[row[order], col[order]] = p_wall[order, 2]
    meta = {"x0": x0, "y0": y0, "gsd": gsd, "width": W, "height": H}
    return img, relief, np.c_[row, col], meta


def fill_small_holes(
    img: np.ndarray, relief: np.ndarray, close_px: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inpaint pixels without points that lie inside the covered area.

    Coverage is the morphological closing of the hit mask with a ``close_px`` square;
    pixels outside it stay empty. Returns (image, relief, coverage mask).
    """
    hit = np.isfinite(relief)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    cover = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    holes = (cover & ~hit).astype(np.uint8)
    img_f = cv2.inpaint(img, holes, 3, cv2.INPAINT_TELEA)
    rel = np.where(hit, relief, 0).astype(np.float32)
    rel_f = cv2.inpaint(rel, holes, 3, cv2.INPAINT_TELEA).astype(float)
    img_f[~cover] = 0
    rel_f[~cover] = np.nan
    return img_f, rel_f, cover


def relief_image(relief: np.ndarray) -> np.ndarray:
    """Relief map as an 8-bit hillshade-plus-height picture for humans."""
    r = np.nan_to_num(relief, nan=np.nanmedian(relief))
    lo, hi = np.percentile(r, [1, 99])
    h = np.clip((r - lo) / max(hi - lo, 1e-9), 0, 1)
    gy, gx = np.gradient(r)
    shade = np.clip(0.5 - 40 * (gx - gy), 0, 1)
    out = (0.6 * shade + 0.4 * h) * 255
    out[~np.isfinite(relief)] = 0
    return out.astype(np.uint8)


# --------------------------------------------------------------------------- masks
def resolve_masks(
    masks: list[np.ndarray],
    valid: np.ndarray,
    min_area_px: float,
    max_area_px: float,
    contain_frac: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Filter masks and resolve overlaps into one label image (TECH_SPEC 4.6 step 3).

    1. Restrict to ``valid`` pixels; drop masks outside ``[min_area_px, max_area_px]``.
    2. Drop a mask that almost entirely contains a smaller kept mask
       (``|small ∩ big| / |small| >= contain_frac``): the finer level wins.
    3. Each pixel goes to the smallest mask containing it.
    4. Drop labels left smaller than ``min_area_px``; renumber 1..N.

    ``masks`` are modified in place (restricted to ``valid``).

    Returns:
        (int32 label image, counts per step).
    """
    for m in masks:  # in place: the full-size masks are large, avoid a second copy
        np.logical_and(m, valid, out=m)
    ms = masks
    areas = np.array([m.sum() for m in ms])
    keep = [i for i in np.argsort(areas) if min_area_px <= areas[i] <= max_area_px]
    counts = {"raw": len(masks), "area_ok": len(keep)}
    boxes = {}
    for i in keep:
        rr, cc = np.nonzero(ms[i])
        boxes[i] = (rr.min(), rr.max() + 1, cc.min(), cc.max() + 1)
    dropped: set[int] = set()
    for a_pos, i in enumerate(keep):  # i small
        r0, r1, c0, c1 = boxes[i]
        small = ms[i][r0:r1, c0:c1]
        for j in keep[a_pos + 1 :]:  # j larger
            if j in dropped:
                continue
            R0, R1, C0, C1 = boxes[j]
            if R0 > r0 or R1 < r1 or C0 > c0 or C1 < c1:
                continue  # bbox of j does not contain bbox of i
            inter = np.count_nonzero(small & ms[j][r0:r1, c0:c1])
            if inter >= contain_frac * areas[i]:
                dropped.add(j)
    kept = [i for i in keep if i not in dropped]
    counts["after_containment"] = len(kept)
    label = np.zeros(valid.shape, np.int32)
    for n, i in enumerate(sorted(kept, key=lambda k: -areas[k]), start=1):
        label[ms[i]] = n  # largest first, so smaller masks overwrite
    ids, cnt = np.unique(label[label > 0], return_counts=True)
    good = ids[cnt >= min_area_px]
    lut = np.zeros(label.max() + 1, np.int32)
    lut[good] = np.arange(1, len(good) + 1)
    label = lut[label]
    counts["stones"] = len(good)
    return label, counts


def label_colours(label: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random colour per label, black for 0. uint8 HxWx3."""
    rng = np.random.default_rng(seed)
    pal = rng.integers(60, 256, size=(label.max() + 1, 3), dtype=np.uint8)
    pal[0] = 0
    return pal[label]


def overlay_boundaries(img: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Orthophoto with label boundaries drawn in yellow and unlabelled area dimmed."""
    out = img.copy()
    out[label == 0] = (out[label == 0] * 0.45).astype(np.uint8)
    edge = np.zeros(label.shape, bool)
    edge[:-1] |= label[:-1] != label[1:]
    edge[:, :-1] |= label[:, :-1] != label[:, 1:]
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    out[edge] = (255, 220, 0)
    return out


# --------------------------------------------------------------------------- helpers
def points_to_ortho_labels(p_wall: np.ndarray, labels: np.ndarray, meta: dict) -> np.ndarray:
    """Front-most point's label per orthophoto pixel (grid of ``meta``), gaps closed.

    A 5 px closing fills pixels without a point inside labelled areas.
    """
    g = meta["gsd"]
    col = np.floor((p_wall[:, 0] - meta["x0"]) / g).astype(int)
    row = np.floor((meta["y0"] - p_wall[:, 1]) / g).astype(int)
    ok = (col >= 0) & (col < meta["width"]) & (row >= 0) & (row < meta["height"])
    o = np.argsort(p_wall[ok, 2])
    lab = np.zeros((meta["height"], meta["width"]), np.int32)
    lab[row[ok][o], col[ok][o]] = labels[ok][o]
    closed = cv2.morphologyEx(lab.astype(np.float32), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return np.where(lab == 0, closed.astype(np.int32), lab)


def load_wall(ctx: StageContext) -> tuple[np.ndarray, np.ndarray, dict]:
    """``dense_wall.ply`` points (wall frame), colours [0, 1], and ``transform.json``."""
    fdir = ctx.stage_dir("frame")
    ply, tj = fdir / "dense_wall.ply", fdir / "transform.json"
    if not ply.is_file() or not tj.is_file():
        raise StageError(f"{fdir} incomplete; run `frame` first")
    pcd = o3d.io.read_point_cloud(str(ply))
    return np.asarray(pcd.points), np.asarray(pcd.colors), json.loads(tj.read_text("utf-8"))


def scale_params(seg: dict) -> dict[str, float]:
    """Orthophoto GSD and stone-area bounds, in scene units (the project is relative)."""
    return {
        "gsd": seg["ortho_gsd"],
        "min_area": seg["min_stone_area"],
        "max_area": seg["max_stone_area"],
    }


def apply_exclusion(
    labels: np.ndarray, inside: np.ndarray, min_inside: float
) -> tuple[np.ndarray, dict[str, int]]:
    """Drop labels mostly inside an exclusion, clear excluded elements, renumber 1..N.

    Works on any label array (a raster or one label per point) with a boolean ``inside``
    of the same shape. A label is dropped when at least ``min_inside`` of its elements
    are inside.
    """
    n = int(labels.max()) if labels.size else 0
    tot = np.bincount(labels.ravel(), minlength=n + 1)
    ins = np.bincount(labels[inside].ravel(), minlength=n + 1)
    ids = np.nonzero(tot[1:])[0] + 1
    keep = ids[ins[ids] < min_inside * tot[ids]]
    lut = np.zeros(n + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    out = lut[labels]
    out[inside] = 0
    return out, {
        "stones_before": len(ids),
        "stones_removed": len(ids) - len(keep),
        "stones_clipped": int(np.count_nonzero(ins[keep])),
    }


def exclusion_mask(ctx: StageContext, img: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Boolean mask of the exclusion polygons on the grid of orthophoto ``img``.

    Returns (None, info) when ``segment.exclude_file`` is null or the file is missing.
    """
    from stonetwin.annotation import shapes_on_current_grid

    name = ctx.cfg["segment"].get("exclude_file")
    if not name or not ctx.cfg.get("annotations_dir"):
        return None, {"file": None}
    path = ctx.root / ctx.cfg["annotations_dir"] / name
    if not path.is_file():
        return None, {"file": None, "missing": path.relative_to(ctx.root).as_posix()}
    ann = json.loads(path.read_text(encoding="utf-8"))
    ann["shapes"] = [s for s in ann["shapes"] if s["label"] == "exclude" and len(s["points"]) >= 3]
    shapes, reg = shapes_on_current_grid(ann, path.parent / "ortho_ref.jpg", img)
    mask = np.zeros(img.shape[:2], np.uint8)
    for sh in shapes:
        cv2.fillPoly(mask, [np.round(np.asarray(sh["points"])).astype(np.int32)], 1)
    return mask.astype(bool), {
        "file": path.relative_to(ctx.root).as_posix(),
        "n_polygons": len(shapes),
        "polygons": [sh.get("description") or sh["label"] for sh in shapes],
        "area_px": int(mask.sum()),
        "registration": reg,
    }


# --------------------------------------------------------------------------- ortho
def run_ortho(
    ctx: StageContext,
    p_wall: np.ndarray,
    rgb: np.ndarray,
    tf: dict,
    excl_px: np.ndarray | None = None,
) -> dict:
    """Orthophoto baseline. Writes ``06_segment/ortho/``; returns metrics."""
    from stonetwin.sam import generate, load_generator

    seg = ctx.cfg["segment"]
    sp = scale_params(seg)
    out = ctx.out_dir / "ortho"
    out.mkdir(parents=True, exist_ok=True)

    img, relief, pix, meta = rasterize(p_wall, rgb, sp["gsd"])
    close_px = int(seg["ortho_hole_close_px"])
    img, relief, cover = fill_small_holes(img, relief, close_px)
    meta.update(
        {"units": tf["units"], "metric": tf["metric"], "frame": "wall", "hole_close_px": close_px}
    )
    Image.fromarray(img).save(out / "ortho.png")
    Image.fromarray(relief_image(relief)).save(out / "relief.png")
    np.save(out / "relief.npy", relief.astype(np.float32))
    (out / "ortho_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    t0 = time.perf_counter()
    gen = load_generator(ctx.cfg)
    records = generate(gen, img)
    t_sam = time.perf_counter() - t0
    px_area = sp["gsd"] ** 2
    masks = [r.pop("segmentation") for r in records]
    del records
    label, counts = resolve_masks(
        masks,
        cover,
        sp["min_area"] / px_area,
        sp["max_area"] / px_area,
        seg["contain_frac"],
    )
    excl_info = None
    if excl_px is not None:
        label, excl_info = apply_exclusion(label, excl_px, seg["exclude_min_inside"])
        counts["stones"] = int(label.max())
    Image.fromarray(label.astype(np.uint16)).save(out / "labels.png")
    Image.fromarray(label_colours(label, int(ctx.cfg["seed"]))).save(out / "labels_vis.png")
    Image.fromarray(overlay_boundaries(img, label)).save(out / "labels_overlay.png")

    pts_labels = label[pix[:, 0], pix[:, 1]].astype(np.int32)
    np.save(out / "points_labels.npy", pts_labels)
    return {
        "image_size": [meta["width"], meta["height"]],
        "gsd": sp["gsd"],
        "units": tf["units"],
        "coverage_fraction": float(cover.mean()),
        "masks": counts,
        "n_stones": int(counts["stones"]),
        "labelled_point_fraction": float((pts_labels > 0).mean()),
        "labelled_area_fraction": float((label > 0).sum() / cover.sum()),
        "sam_seconds": round(t_sam, 1),
        "exclusion": excl_info,
    }


# --------------------------------------------------------------------------- stage
def run(ctx: StageContext) -> None:
    """Run every method in ``segment.methods``; each writes its own sub-folder.

    Raises:
        AcceptanceError: a method produced no stones.
    """
    seg = dict(ctx.cfg["segment"])
    ctx.params = seg
    ctx.inputs = {"cloud": "04_frame/dense_wall.ply", "transform": "04_frame/transform.json"}
    p_wall, rgb, tf = load_wall(ctx)
    ctx.metrics = {"metric": tf["metric"], "n_points": int(len(p_wall))}

    # Exclusion polygons on the orthophoto grid, and per point (D-021).
    img, relief, pix, _ = rasterize(p_wall, rgb, scale_params(seg)["gsd"])
    img, _, _ = fill_small_holes(img, relief, int(seg["ortho_hole_close_px"]))
    del relief
    excl_px, excl = exclusion_mask(ctx, img)
    excl_pt = excl_px[pix[:, 0], pix[:, 1]] if excl_px is not None else None
    if excl_px is not None:
        reg = excl["registration"]
        if reg.get("n_inliers", np.inf) < ctx.cfg["evaluate"]["min_registration_inliers"]:
            raise AcceptanceError(f"exclusion registration has only {reg['n_inliers']} inliers")
        excl["point_fraction"] = float(excl_pt.mean())
        ctx.inputs["exclusions"] = excl["file"]
    ctx.metrics["exclusions"] = excl
    del img

    for method in seg["methods"]:
        if method == "ortho":
            m = run_ortho(ctx, p_wall, rgb, tf, excl_px)
        elif method == "multiview":
            from stonetwin.stages.segment_multiview import run_multiview

            m = run_multiview(ctx, p_wall, rgb, tf)
            if excl_pt is not None:
                f = ctx.out_dir / "multiview" / "points_labels.npy"
                lab, m["exclusion"] = apply_exclusion(
                    np.load(f), excl_pt, seg["exclude_min_inside"]
                )
                np.save(f, lab)
                m["n_stones"] = int(lab.max())
                m["labelled_point_fraction"] = float((lab > 0).mean())
        else:
            raise StageError(f"unknown segmentation method '{method}'")
        ctx.metrics[method] = m
        if m["n_stones"] == 0:
            raise AcceptanceError(f"method {method} found no stones")
