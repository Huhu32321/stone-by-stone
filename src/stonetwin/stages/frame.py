"""Stage ``frame``: wall plane, wall frame, scale and region of interest.

Frames:
    ``world``: the SfM frame (arbitrary origin, orientation, scale).
    ``wall``: x along the wall (horizontal), y up, z out of the wall toward the
    cameras; origin at the plane-inlier centroid. Units are metres when
    ``scale_refs.json`` exists (``metric: true``), otherwise scene units (scale 1).

``p_wall = T_wall_from_world @ [p_world, 1]`` with
``T_wall_from_world = [[s R, -s R o], [0, 1]]``, ``R`` rows = wall axes in world,
``o`` = origin in world, ``s`` = metres per scene unit (1 when non-metric).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

from stonetwin.stage import AcceptanceError, StageContext, StageError


# --------------------------------------------------------------------------- geometry
def fit_plane_lsq(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through ``p`` (N, 3). Returns (unit normal, centroid)."""
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    return vt[2], c


def fit_plane_ransac(
    p: np.ndarray, threshold: float, iterations: int = 2000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC plane, refined by least squares on the inliers.

    Returns:
        (unit normal, inlier centroid, boolean inlier mask). Units follow ``p``.
    """
    rng = np.random.default_rng(seed)
    best = None
    best_count = -1
    for _ in range(iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        count = int(np.count_nonzero(np.abs((p - a) @ n) < threshold))
        if count > best_count:
            best, best_count = (n, a), count
    if best is None:
        raise StageError("plane RANSAC found no non-degenerate sample")
    n, a = best
    inl = np.abs((p - a) @ n) < threshold
    for _ in range(3):  # refine, re-select inliers against the refined plane
        n, c = fit_plane_lsq(p[inl])
        inl = np.abs((p - c) @ n) < threshold
    n, c = fit_plane_lsq(p[inl])
    return n, c, inl


def wall_axes(
    normal_world: np.ndarray,
    cam_centers_world: np.ndarray,
    cam_ups_world: np.ndarray,
    origin_world: np.ndarray,
) -> np.ndarray:
    """Rotation ``R_wall_from_world`` (rows are the wall x, y, z axes in world).

    z = plane normal oriented toward the mean camera centre; y = mean camera up-vector
    projected into the plane; x = y cross z (right-handed).
    """
    z = normal_world / np.linalg.norm(normal_world)
    if (cam_centers_world.mean(axis=0) - origin_world) @ z < 0:
        z = -z
    up = cam_ups_world.mean(axis=0)
    y = up - z * (up @ z)
    if np.linalg.norm(y) < 1e-6:
        raise StageError("camera up-vector is parallel to the wall normal")
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def make_T(R: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    """``T_wall_from_world`` from rotation rows ``R``, world origin and scale."""
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = -scale * R @ origin
    return T


def apply_T(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 4x4 similarity to points (N, 3)."""
    return p @ T[:3, :3].T + T[:3, 3]


def triangulate(P_list: list[np.ndarray], x_list: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation from normalised image points.

    Args:
        P_list: 3x4 ``[R | t]`` cam-from-world matrices (normalised coordinates).
        x_list: undistorted normalised image points ``(x, y)``.
    Returns:
        The point in world coordinates.
    """
    A = []
    for P, (u, v) in zip(P_list, x_list, strict=True):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(A))
    X = vt[-1]
    return X[:3] / X[3]


# --------------------------------------------------------------------------- inputs
def _cameras(ctx: StageContext) -> dict[str, Any]:
    path = ctx.stage_dir("sfm") / "cameras.json"
    if not path.is_file():
        raise StageError(f"{path} missing; run `sfm` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _cam_arrays(cams: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera centres, up-vectors and forward-vectors in world (OpenCV: up = -y_cam)."""
    T = np.array([im["T_world_from_cam"] for im in cams["images"]])
    return T[:, :3, 3], -T[:, :3, 1], T[:, :3, 2]


def _scale_from_refs(path: Path, cams: dict) -> tuple[float, list[dict]]:
    """Scale (metres per scene unit) as the median ratio over references."""
    refs = json.loads(path.read_text(encoding="utf-8"))["references"]
    by_name = {im["name"]: im for im in cams["images"]}
    rows, ratios = [], []
    for ref in refs:
        ends = []
        for key in ("p1", "p2"):
            Ps, xs = [], []
            for ob in ref["observations"]:
                im = by_name.get(ob["image"])
                if im is None:
                    continue
                cam = cams["cameras"][str(im["camera_id"])]
                K = np.array(cam["K"])
                dist = np.array(cam["dist"] or [0, 0, 0, 0], float)
                xn = cv2.undistortPoints(np.array([[ob[key]]], float), K, dist).reshape(2)
                Ps.append(np.linalg.inv(np.array(im["T_world_from_cam"]))[:3])
                xs.append(xn)
            if len(Ps) < 2:
                raise StageError(
                    f"scale reference '{ref['name']}' has fewer than two registered views"
                )
            ends.append(triangulate(Ps, xs))
        d_scene = float(np.linalg.norm(ends[0] - ends[1]))
        if ref["distance_m"] <= 0 or d_scene <= 0:
            raise StageError(f"scale reference '{ref['name']}' has a non-positive distance")
        ratios.append(ref["distance_m"] / d_scene)
        rows.append({"name": ref["name"], "distance_m": ref["distance_m"], "d_scene": d_scene})
    s = float(np.median(ratios))
    for r in rows:
        r["residual_m"] = s * r["d_scene"] - r["distance_m"]
    return s, rows


def _roi_hull_wall(cams: dict, T: np.ndarray, frac: float) -> np.ndarray:
    """Wall-plane points (M, 2) where the central ``frac`` of each image hits z = 0."""
    pts = []
    for im in cams["images"]:
        cam = cams["cameras"][str(im["camera_id"])]
        K = np.array(cam["K"])
        w, h = cam["width"], cam["height"]
        lo, hi = (1 - frac) / 2, (1 + frac) / 2
        pix = np.array(
            [[w * a, h * b, 1] for a in (lo, hi) for b in (lo, hi)] + [[w / 2, h / 2, 1]]
        )
        rays_cam = pix @ np.linalg.inv(K).T
        Twc = np.array(im["T_world_from_cam"])
        o_wall = apply_T(T, Twc[:3, 3][None])[0]
        d_wall = (rays_cam @ Twc[:3, :3].T) @ T[:3, :3].T
        t = -o_wall[2] / d_wall[:, 2]
        ok = t > 0
        pts.append((o_wall[None] + t[ok, None] * d_wall[ok])[:, :2])
    return np.concatenate(pts)


# --------------------------------------------------------------------------- stage
def run(ctx: StageContext) -> None:
    """Fit the wall frame, apply scale, crop ``dense_wall.ply`` (wall frame).

    Raises:
        AcceptanceError: too few plane inliers, or ``frame.require_metric`` is true
            and no scale references exist.
    """
    fcfg = dict(ctx.cfg["frame"])
    ctx.params = fcfg
    dense = ctx.stage_dir("dense") / "dense.ply"
    if not dense.is_file():
        raise StageError(f"{dense} missing; run `dense` first")
    cams = _cameras(ctx)
    centers, ups, _ = _cam_arrays(cams)
    pcd = o3d.io.read_point_cloud(str(dense))
    p_world = np.asarray(pcd.points)
    seed = int(ctx.cfg["seed"])

    refs_path = ctx.root / ctx.cfg["meta_dir"] / "scale_refs.json"
    ctx.inputs = {
        "dense": "02_dense/dense.ply",
        "cameras": "01_sfm/cameras.json",
        "scale_refs": refs_path.relative_to(ctx.root).as_posix() if refs_path.is_file() else None,
    }
    if refs_path.is_file():
        scale, ref_rows = _scale_from_refs(refs_path, cams)
        metric = True
    else:
        if fcfg["require_metric"]:
            raise AcceptanceError(f"frame.require_metric is true but {refs_path} is missing")
        scale, ref_rows, metric = 1.0, [], False

    geometry = ctx.cfg.get("geometry", "planar")
    plane_info: dict[str, Any] = {}
    if geometry == "planar":
        rng = np.random.default_rng(seed)
        sub = p_world[rng.choice(len(p_world), min(len(p_world), 300_000), replace=False)]
        n, _, inl_sub = fit_plane_ransac(sub, fcfg["plane_ransac_threshold"], seed=seed)
        inl = np.abs((p_world - sub[inl_sub].mean(0)) @ n) < fcfg["plane_ransac_threshold"]
        n, origin = fit_plane_lsq(p_world[inl])
        inlier_fraction = float(inl.mean())
        if inlier_fraction < fcfg["min_inlier_fraction"]:
            raise AcceptanceError(
                f"wall plane inlier fraction {inlier_fraction:.3f} < {fcfg['min_inlier_fraction']}"
            )
        R = wall_axes(n, centers, ups, origin)
        resid = (p_world[inl] - origin) @ R[2]
        plane_info = {
            "frame": "colmap_world",
            "normal_world": R[2].tolist(),
            "point_world": origin.tolist(),
            "inlier_fraction": inlier_fraction,
            "inlier_rms": float(np.sqrt(np.mean(resid**2))),
            "units": "scene",
        }
    else:  # freeform: gravity alignment only
        origin = p_world.mean(axis=0)
        fwd = origin - centers.mean(0)
        R = wall_axes(-fwd, centers, ups, origin)

    T = make_T(R, origin, scale)
    p_wall = apply_T(T, p_world)

    keep = np.ones(len(p_wall), bool)
    if geometry == "planar":
        band = np.array([-fcfg["roi_band_back"], fcfg["roi_band_front"]]) * scale
        keep &= (p_wall[:, 2] >= band[0]) & (p_wall[:, 2] <= band[1])
        hull_pts = _roi_hull_wall(cams, T, fcfg["roi_image_fraction"])
        keep &= Delaunay(hull_pts).find_simplex(p_wall[:, :2]) >= 0
    if keep.sum() == 0:
        raise AcceptanceError("wall region of interest is empty")

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(p_wall[keep])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep])
    if pcd.has_normals():
        nrm = np.asarray(pcd.normals)[keep] @ R.T
        out.normals = o3d.utility.Vector3dVector(nrm)
    o3d.io.write_point_cloud(str(ctx.out_dir / "dense_wall.ply"), out)

    units = "m" if metric else "scene"
    transform = {
        "T_wall_from_world": T.tolist(),
        "R_wall_from_world": R.tolist(),
        "origin_world": origin.tolist(),
        "scale": scale,
        "metric": metric,
        "units": units,
        "geometry": geometry,
        "scale_refs": ref_rows,
        "convention": "p_wall = T_wall_from_world @ [p_world, 1]; x along wall, y up, "
        "z out of wall toward cameras",
    }
    (ctx.out_dir / "transform.json").write_text(json.dumps(transform, indent=2), "utf-8")
    if plane_info:
        (ctx.out_dir / "wall_plane.json").write_text(json.dumps(plane_info, indent=2), "utf-8")

    pk = p_wall[keep]
    ctx.metrics = {
        "metric": metric,
        "units": units,
        "scale": scale,
        "scale_residuals_m": [r["residual_m"] for r in ref_rows],
        "n_points_in": int(len(p_wall)),
        "n_points_wall": int(keep.sum()),
        "wall_extent": (pk.max(0) - pk.min(0)).tolist(),
        **{k: plane_info[k] for k in ("inlier_fraction", "inlier_rms") if k in plane_info},
    }
