"""Stage ``attributes``: one record per stone (TECH_SPEC 4.7).

Input: ``04_frame/dense_wall.ply`` (wall frame: x along the wall, y up, z out of the
wall) and ``06_segment/<method>/points_labels.npy``. Lengths are metres when
``transform.json`` says ``metric: true``, otherwise scene units; ``metric`` is written
at the top of ``stones.json``.

``roughness_rms``, ``recession`` and ``colour_delta_e`` are condition proxies: simple
geometric and colour measurements. They are not deterioration classes.
"""

from __future__ import annotations

import csv
import json
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.color import deltaE_ciede2000, rgb2lab

from stonetwin.stage import AcceptanceError, StageContext, StageError
from stonetwin.stages.segment import load_wall, scale_params

PROXY_NOTE = (
    "roughness_rms, recession and colour_delta_e are condition proxies (geometric and "
    "colour measurements). They are not validated deterioration classifications."
)


def footprint(xy: np.ndarray, gsd: float, close_px: int) -> tuple[float, np.ndarray]:
    """Face area and outline of a stone's points projected on the wall plane.

    The points are rasterised at ``gsd``, closed with a ``close_px`` disc, and the
    largest connected region is taken. Its pixel count is the area; its outer contour
    (in wall coordinates) is a concave hull of the points.

    Returns:
        (area in units^2, outline (M, 2) in wall x, y).
    """
    x0, y0 = xy.min(0) - 2 * close_px * gsd
    c = np.floor((xy[:, 0] - x0) / gsd).astype(int)
    r = np.floor((xy[:, 1] - y0) / gsd).astype(int)
    img = np.zeros((r.max() + 2 * close_px + 1, c.max() + 2 * close_px + 1), np.uint8)
    img[r, c] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    if n < 2:
        return 0.0, np.zeros((0, 2))
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (cc == big).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    outline = np.c_[x0 + (cnt[:, 0] + 0.5) * gsd, y0 + (cnt[:, 1] + 0.5) * gsd]
    return float(mask.sum()) * gsd**2, outline


def min_rect_extents(outline: np.ndarray) -> tuple[float, float]:
    """(width, height) of the minimum-area rectangle around ``outline``.

    ``width`` is the side closer to the wall's horizontal (x) axis.
    """
    (_, _), (a, b), ang = cv2.minAreaRect(outline.astype(np.float32))
    # OpenCV: side `a` is at angle `ang` (degrees) from the x axis.
    ang = abs(((ang + 90) % 180) - 90)
    return (float(a), float(b)) if ang <= 45 else (float(b), float(a))


def roughness(p: np.ndarray) -> float:
    """RMS distance of points to their own least-squares plane."""
    q = p - p.mean(0)
    _, s, _ = np.linalg.svd(q, full_matrices=False)
    return float(s[2] / np.sqrt(len(p)))


def stone_records(
    p_wall: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    gsd: float,
    close_px: int,
    k_neighbours: int,
    min_points: int,
    radius: float,
) -> list[dict[str, Any]]:
    """Attributes for every label > 0 with at least ``min_points`` points.

    ``radius`` bounds the neighbour stones that define the local wall surface for
    ``recession`` (in the cloud's units).
    """
    lab_all = rgb2lab(np.clip(rgb, 0, 1)[None])[0]
    lab_ref = np.median(lab_all, axis=0)
    ids = [int(i) for i in np.unique(labels) if i > 0 and (labels == i).sum() >= min_points]
    recs: list[dict[str, Any]] = []
    for i in ids:
        m = labels == i
        p, lab = p_wall[m], lab_all[m]
        area, outline = footprint(p[:, :2], gsd, close_px)
        w, h = min_rect_extents(outline) if len(outline) >= 3 else (0.0, 0.0)
        mean_lab = lab.mean(0)
        recs.append(
            {
                "id": i,
                "n_points": int(m.sum()),
                "centroid_wall": p.mean(0).tolist(),
                "face_area": area,
                "width": w,
                "height": h,
                "aspect_ratio": w / h if h > 0 else float("nan"),
                "roughness_rms": roughness(p),
                "_z_median": float(np.median(p[:, 2])),
                "colour_lab": mean_lab.tolist(),
                "colour_std": float(lab[:, 0].std()),
                "colour_delta_e": float(deltaE_ciede2000(mean_lab[None], lab_ref[None])[0]),
                "outline_wall": outline.tolist(),
            }
        )
    # Recession: own median z minus the median of the median z of neighbouring stones,
    # neighbours = all stones whose centroid lies within `radius` (at least k nearest).
    # The radius must exceed the size of recessed areas, or a stone inside a cavity is
    # compared only with other cavity stones (D-018).
    cxy = np.array([r["centroid_wall"][:2] for r in recs])
    zmed = np.array([r["_z_median"] for r in recs])
    tree = cKDTree(cxy)
    k = min(k_neighbours + 1, len(recs))
    _, nn = tree.query(cxy, k=k)
    for j, r in enumerate(recs):
        near = set(tree.query_ball_point(cxy[j], radius)) | set(np.atleast_1d(nn[j]).tolist())
        neigh = sorted(n for n in near if n != j)
        r["recession"] = float(zmed[j] - np.median(zmed[neigh])) if neigh else 0.0
        r["n_neighbours"] = len(neigh)
        del r["_z_median"]
    return recs


FIELDS = [
    "id",
    "n_points",
    "centroid_wall",
    "face_area",
    "width",
    "height",
    "aspect_ratio",
    "roughness_rms",
    "recession",
    "colour_lab",
    "colour_std",
    "colour_delta_e",
    "outline_wall",
]


def validate(recs: list[dict[str, Any]]) -> list[str]:
    """Problems with the records: missing fields or non-finite numbers."""
    problems = []
    for r in recs:
        for f in FIELDS:
            if f not in r:
                problems.append(f"stone {r.get('id')}: missing {f}")
                continue
            vals = np.asarray(r[f], float).ravel()
            if f != "outline_wall" and not np.all(np.isfinite(vals)):
                problems.append(f"stone {r['id']}: non-finite {f}")
    return problems


def run(ctx: StageContext) -> None:
    """Write ``stones.json`` and ``stones.csv`` for ``attributes.method``.

    Raises:
        AcceptanceError: a record is missing a field or has a NaN.
    """
    acfg = dict(ctx.cfg["attributes"])
    method = acfg["method"]
    ctx.params = acfg
    lab_path = ctx.stage_dir("segment") / method / "points_labels.npy"
    if not lab_path.is_file():
        raise StageError(f"{lab_path} missing; run `segment` with method {method}")
    p_wall, rgb, tf = load_wall(ctx)
    labels = np.load(lab_path)
    if len(labels) != len(p_wall):
        raise StageError("points_labels.npy does not match dense_wall.ply")
    ctx.inputs = {
        "cloud": "04_frame/dense_wall.ply",
        "labels": lab_path.relative_to(ctx.scene_dir).as_posix(),
    }
    gsd = scale_params(ctx.cfg["segment"], tf["metric"])["gsd"]
    rad = acfg["recession_radius_m"] if tf["metric"] else acfg["nonmetric"]["recession_radius"]
    recs = stone_records(
        p_wall,
        rgb,
        labels,
        gsd,
        acfg["footprint_close_px"],
        acfg["recession_neighbours"],
        acfg["min_points"],
        rad,
    )
    problems = validate(recs)
    units = tf["units"]
    doc = {
        "metric": tf["metric"],
        "units": units,
        "frame": "wall (x along wall, y up, z out of wall)",
        "method": method,
        "note": PROXY_NOTE,
        "fields": {
            "face_area": f"{units}^2, concave hull of points on the wall plane",
            "width/height": f"{units}, minimum-area rectangle; width is the side nearer horizontal",
            "roughness_rms": f"{units}, RMS residual to the stone's own plane",
            "recession": f"{units}, median z minus median z of {acfg['recession_neighbours']} "
            "nearest stones; negative = set back",
            "colour_lab": "CIELAB mean (D65)",
            "colour_std": "standard deviation of L*",
            "colour_delta_e": "CIEDE2000 between the stone mean and the wall median colour",
        },
        "stones": recs,
    }
    (ctx.out_dir / "stones.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    with (ctx.out_dir / "stones.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "n_points",
                "cx",
                "cy",
                "cz",
                "face_area",
                "width",
                "height",
                "aspect_ratio",
                "roughness_rms",
                "recession",
                "L",
                "a",
                "b",
                "colour_std",
                "colour_delta_e",
            ]
        )
        for r in recs:
            w.writerow(
                [
                    r["id"],
                    r["n_points"],
                    *r["centroid_wall"],
                    r["face_area"],
                    r["width"],
                    r["height"],
                    r["aspect_ratio"],
                    r["roughness_rms"],
                    r["recession"],
                    *r["colour_lab"],
                    r["colour_std"],
                    r["colour_delta_e"],
                ]
            )

    rec = np.array([r["recession"] for r in recs])
    ctx.metrics = {
        "method": method,
        "metric": tf["metric"],
        "units": units,
        "n_stones": len(recs),
        "n_problems": len(problems),
        "problems": problems[:20],
        "face_area_median": float(np.median([r["face_area"] for r in recs])),
        "roughness_rms_median": float(np.median([r["roughness_rms"] for r in recs])),
        "recession_median": float(np.median(rec)),
        "recession_p05": float(np.percentile(rec, 5)),
        "colour_delta_e_median": float(np.median([r["colour_delta_e"] for r in recs])),
    }
    if problems:
        raise AcceptanceError(f"{len(problems)} invalid stone records, e.g. {problems[0]}")
