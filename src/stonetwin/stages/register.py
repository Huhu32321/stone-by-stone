"""Stage ``register``: similarity alignment of two independent half-reconstructions.

The usable images are split into two disjoint subsets by capture order. Each subset is
reconstructed on its own (COLMAP SfM + MVS), so the two dense clouds live in unrelated
gauges ``a`` and ``b`` (origin, orientation and scale all differ). The stage estimates
``T_a_from_b`` (written as ``T_b_to_a.json``), a 4x4 similarity with
``p_a = s R p_b + t``.

This is one capture session split in two. It tests the estimator, not change over time.

Coarse: each cloud is normalised (centred on its median, divided by its median radius),
then FPFH features are matched with RANSAC estimating a similarity.
Fine: point-to-plane ICP that estimates scale, rotation and translation jointly
(linearised 7-DoF step per iteration), until scale and rotation stop changing (D-015).

Reference: the full-scene SfM model contains every image, so the cameras of each half
give an independent ``T_full_from_a`` and ``T_full_from_b`` (Umeyama on camera centres).
``inv(T_full_from_a) @ T_full_from_b`` is the reference against which the cloud-based
estimate is compared. The cloud-based estimate never sees camera poses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree

from stonetwin.stage import AcceptanceError, StageContext, StageError
from stonetwin.stages.dense import colmap_mvs, median_nn_spacing
from stonetwin.stages.sfm import colmap_sfm, export_cameras, largest_model
from stonetwin.tools import colmap_version, find_tool

reg = o3d.pipelines.registration


# --------------------------------------------------------------------------- maths
def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity with ``dst ≈ s R src + t`` (Umeyama 1991).

    Args:
        src, dst: corresponding points, shape (N, 3), N >= 3, same units per array.
    Returns:
        (s, R, t). ``s`` is 1 when ``with_scale`` is false.
    """
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("need at least 3 corresponding points of equal shape")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def sim_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 matrix of ``p -> s R p + t``."""
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def decompose(T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Inverse of :func:`sim_matrix` for a similarity matrix."""
    M = T[:3, :3]
    s = float(np.cbrt(np.linalg.det(M)))
    return s, M / s, T[:3, 3].copy()


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def apply_T(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to points (N, 3)."""
    return p @ T[:3, :3].T + T[:3, 3]


def split_images(names_in_capture_order: list[str], mode: str) -> tuple[list[str], list[str]]:
    """Disjoint subsets ``(a, b)``.

    ``odd_even``: a = even positions, b = odd positions.
    ``thirds``: a = positions 0, 1 mod 3 (two thirds), b = positions 2 mod 3.
    """
    n = names_in_capture_order
    if mode == "odd_even":
        return n[0::2], n[1::2]
    if mode == "thirds":
        return [x for i, x in enumerate(n) if i % 3 != 2], n[2::3]
    raise StageError(f"register.split must be odd_even or thirds, got {mode}")


def normaliser(p: np.ndarray) -> np.ndarray:
    """4x4 similarity taking ``p`` to unit median radius around its median point."""
    c = np.median(p, axis=0)
    r = float(np.median(np.linalg.norm(p - c, axis=1)))
    return sim_matrix(1.0 / r, np.eye(3), -c / r)


# --------------------------------------------------------------------------- alignment
def _pcd(p: np.ndarray, voxel: float, normal_radius: float) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)).voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    return pc


def coarse_align(
    pa_n: np.ndarray, pb_n: np.ndarray, rc: dict, seed: int
) -> tuple[np.ndarray, dict]:
    """FPFH + RANSAC similarity on normalised clouds. Returns (T_an_from_bn, info)."""
    v = rc["voxel"]
    a = _pcd(pa_n, v, v * rc["normal_radius_factor"])
    b = _pcd(pb_n, v, v * rc["normal_radius_factor"])
    kd = o3d.geometry.KDTreeSearchParamHybrid(radius=v * rc["fpfh_radius_factor"], max_nn=100)
    fa = reg.compute_fpfh_feature(a, kd)
    fb = reg.compute_fpfh_feature(b, kd)
    dist = v * rc["ransac_distance_factor"]
    o3d.utility.random.seed(seed)
    res = reg.registration_ransac_based_on_feature_matching(
        b,
        a,
        fb,
        fa,
        True,
        dist,
        reg.TransformationEstimationPointToPoint(with_scaling=True),
        3,
        [
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        reg.RANSACConvergenceCriteria(rc["ransac_max_iterations"], rc["ransac_confidence"]),
    )
    info = {
        "fitness": float(res.fitness),
        "inlier_rmse_normalised": float(res.inlier_rmse),
        "n_points_a": len(a.points),
        "n_points_b": len(b.points),
    }
    return np.asarray(res.transformation), info


def rodrigues(w: np.ndarray) -> np.ndarray:
    """Rotation matrix for the rotation vector ``w`` (radians)."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def p2plane_similarity_step(
    src: np.ndarray, dst: np.ndarray, n_dst: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """One linearised 7-DoF point-to-plane step.

    Minimises ``sum (n . (s R (x - c) + c + v - y))^2`` over log-scale, rotation vector
    and translation ``v``, about the source centroid ``c`` (for conditioning).

    Returns:
        (4x4 update applied to ``src``, |log-scale step|, rotation step in degrees).
    """
    c = src.mean(0)
    q = src - c
    J = np.c_[np.einsum("ij,ij->i", n_dst, q), np.cross(q, n_dst), n_dst]
    r = -np.einsum("ij,ij->i", n_dst, src - dst)
    x, *_ = np.linalg.lstsq(J, r, rcond=None)
    ds, w, v = x[0], x[1:4], x[4:7]
    s, R = float(np.exp(ds)), rodrigues(w)
    T = sim_matrix(s, R, c + v - s * R @ c)
    return T, abs(float(ds)), float(np.degrees(np.linalg.norm(w)))


def fine_align(
    pa_n: np.ndarray, pb_n: np.ndarray, T0: np.ndarray, rc: dict
) -> tuple[np.ndarray, dict]:
    """Point-to-plane ICP estimating a similarity jointly (scale, rotation, translation).

    Each iteration pairs every point of B with its nearest point of A within
    ``icp_max_distance_factor * icp_voxel`` and takes one linearised 7-DoF
    point-to-plane step. Normalised units. Converged when one step changes scale by less
    than ``converge_scale`` and rotation by less than ``converge_rot_deg``.
    """
    v = rc["icp_voxel"]
    a = _pcd(pa_n, v, v * rc["normal_radius_factor"])
    b = _pcd(pb_n, v, v * rc["normal_radius_factor"])
    pa, na, pb = np.asarray(a.points), np.asarray(a.normals), np.asarray(b.points)
    tree = cKDTree(pa)
    d = v * rc["icp_max_distance_factor"]
    T = T0.copy()
    history = []
    converged = False
    for it in range(rc["icp_max_iterations"]):
        x = apply_T(T, pb)
        dist, idx = tree.query(x, k=1, distance_upper_bound=d, workers=-1)
        ok = np.isfinite(dist)
        if ok.sum() < 7:
            raise AcceptanceError(f"ICP iteration {it}: fewer than 7 correspondences")
        dT, ds, drot = p2plane_similarity_step(x[ok], pa[idx[ok]], na[idx[ok]])
        T = dT @ T
        history.append(
            {
                "iteration": it,
                "fitness": float(ok.mean()),
                "rmse": float(np.sqrt(np.mean(dist[ok] ** 2))),
                "scale_step": ds,
                "rotation_step_deg": drot,
            }
        )
        if ds < rc["converge_scale"] and drot < rc["converge_rot_deg"]:
            converged = True
            break
    x = apply_T(T, pb)
    dist, _ = tree.query(x, k=1, distance_upper_bound=d, workers=-1)
    ok = np.isfinite(dist)
    return T, {
        "converged": converged,
        "iterations": len(history),
        "history": history[:: max(1, len(history) // 25)] + history[-1:],
        "fitness": float(ok.mean()),
        "inlier_rmse_normalised": float(np.sqrt(np.mean(dist[ok] ** 2))),
        "max_correspondence_distance_normalised": d,
    }


# --------------------------------------------------------------------------- stage
def _reconstruct(ctx: StageContext, tag: str, names: list[str], colmap: Path) -> dict:
    """SfM + MVS for one subset under ``05_register/<tag>/``."""
    work = ctx.out_dir / tag
    raw = ctx.root / ctx.cfg["raw_dir"]
    seed = int(ctx.cfg["seed"])
    sparse = colmap_sfm(colmap, raw, names, work / "colmap", ctx.cfg["sfm"], seed, work / "sfm.log")
    model_dir, rec, n_models = largest_model(sparse)
    cams = export_cameras(rec, model_dir.relative_to(work).as_posix())
    (work / "cameras.json").write_text(json.dumps(cams, indent=1), encoding="utf-8")
    n_reg = len(cams["images"])
    info = {
        "n_images": len(names),
        "n_registered": n_reg,
        "registered_fraction": n_reg / len(names),
        "mean_reproj_error_px": float(rec.compute_mean_reprojection_error()),
        "n_models": n_models,
    }
    acc = ctx.cfg["sfm"]["accept"]
    if info["registered_fraction"] < acc["min_registered_fraction"]:
        raise AcceptanceError(
            f"subset {tag} registered {n_reg}/{len(names)} images; switch register.split "
            "to 'thirds' (TASKS M3) and log it"
        )
    fused, n_depth = colmap_mvs(
        colmap, raw, model_dir, work / "mvs", ctx.cfg["dense"], seed, work / "mvs.log"
    )
    info["n_depth_maps"] = n_depth
    info["cameras"] = cams
    info["fused"] = fused
    return info


def _camera_reference(full: dict, cams: dict) -> np.ndarray:
    """``T_full_from_subset`` from camera centres (Umeyama, with scale)."""
    by_name = {im["name"]: im["center_world"] for im in full["images"]}
    sub = [
        (im["center_world"], by_name[im["name"]]) for im in cams["images"] if im["name"] in by_name
    ]
    src = np.array([s for s, _ in sub])
    dst = np.array([d for _, d in sub])
    return sim_matrix(*umeyama(src, dst))


def _c2c_figure(p_b_in_a: np.ndarray, signed: np.ndarray, lim: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = p_b_in_a - p_b_in_a.mean(0)
    _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 50_000)], full_matrices=False)
    uv = c @ vt[:2].T
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    sc = ax.scatter(
        uv[:, 0], uv[:, 1], c=signed, s=0.2, cmap="coolwarm", vmin=-lim, vmax=lim, rasterized=True
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.colorbar(sc, ax=ax, shrink=0.7, label="signed distance to cloud A (A's scene units)")
    ax.set_title("Half B aligned to half A, coloured by signed cloud-to-cloud distance")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def run(ctx: StageContext) -> None:
    """Split, reconstruct both halves, estimate ``T_b_to_a``, report statistics.

    Raises:
        AcceptanceError: a half fails SfM acceptance, or ICP does not converge.
    """
    rc = dict(ctx.cfg["register"])
    ctx.params = rc
    seed = int(ctx.cfg["seed"])
    colmap = find_tool("colmap", ctx.cfg)
    ctx.tools = {"colmap": colmap_version(colmap), "open3d": o3d.__version__}

    man = pd.read_csv(ctx.stage_dir("ingest") / "manifest.csv")
    man = man[man["use"].astype(bool)].sort_values("order")
    names_a, names_b = split_images(man["file"].tolist(), rc["split"])
    ctx.inputs = {
        "manifest": "00_ingest/manifest.csv",
        "split": rc["split"],
        "images_a": names_a,
        "images_b": names_b,
    }

    info_a = _reconstruct(ctx, "a", names_a, colmap)
    info_b = _reconstruct(ctx, "b", names_b, colmap)
    pa = np.asarray(o3d.io.read_point_cloud(str(info_a["fused"])).points)
    pcd_b = o3d.io.read_point_cloud(str(info_b["fused"]))
    pb = np.asarray(pcd_b.points)
    o3d.io.write_point_cloud(
        str(ctx.out_dir / "subset_a.ply"), o3d.io.read_point_cloud(str(info_a["fused"]))
    )
    o3d.io.write_point_cloud(str(ctx.out_dir / "subset_b.ply"), pcd_b)

    Na, Nb = normaliser(pa), normaliser(pb)
    pa_n, pb_n = apply_T(Na, pa), apply_T(Nb, pb)
    T0_n, coarse = coarse_align(pa_n, pb_n, rc, seed)
    T_n, fine = fine_align(pa_n, pb_n, T0_n, rc)
    T = np.linalg.inv(Na) @ T_n @ Nb  # T_a_from_b in native units
    T_coarse = np.linalg.inv(Na) @ T0_n @ Nb
    s, R, t = decompose(T)
    s_a = 1.0 / Na[0, 0]  # A's normalisation radius, to convert normalised lengths

    # Cloud-to-cloud distances, B aligned into A, in A's scene units.
    pb_in_a = apply_T(T, pb)
    tree = cKDTree(pa)
    dist, idx = tree.query(pb_in_a, k=1, workers=-1)
    a_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pa))
    a_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    na = np.asarray(a_pcd.normals)[idx]
    signed = np.einsum("ij,ij->i", pb_in_a - pa[idx], na)
    overlap_d = fine["max_correspondence_distance_normalised"] * s_a
    ov = dist < overlap_d
    spacing_a = median_nn_spacing(pa, seed=seed)
    o3d.io.write_point_cloud(
        str(ctx.out_dir / "subset_b_in_a.ply"),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pb_in_a)),
    )
    _c2c_figure(
        pb_in_a, signed, float(np.percentile(np.abs(signed[ov]), 95)), ctx.out_dir / "c2c.png"
    )

    # Independent reference from camera poses in the full-scene model.
    full = json.loads((ctx.stage_dir("sfm") / "cameras.json").read_text(encoding="utf-8"))
    T_ref = np.linalg.inv(_camera_reference(full, info_a["cameras"])) @ _camera_reference(
        full, info_b["cameras"]
    )
    s_ref, R_ref, t_ref = decompose(T_ref)
    extent_a = float(np.linalg.norm(np.percentile(pa, 99, 0) - np.percentile(pa, 1, 0)))
    ref = {
        "T_a_from_b": T_ref.tolist(),
        "scale": s_ref,
        "scale_rel_error": s / s_ref - 1,
        "rotation_error_deg": rotation_angle_deg(R @ R_ref.T),
        "translation_error": float(np.linalg.norm(t - t_ref)),
        "translation_error_rel_extent": float(np.linalg.norm(t - t_ref)) / extent_a,
        "coarse_scale_rel_error": decompose(T_coarse)[0] / s_ref - 1,
        "coarse_rotation_error_deg": rotation_angle_deg(decompose(T_coarse)[1] @ R_ref.T),
    }

    out = {
        "T_a_from_b": T.tolist(),
        "convention": "p_a = T_a_from_b @ [p_b, 1] = s R p_b + t; a, b are the native "
        "SfM frames of the two half-reconstructions (non-metric)",
        "scale": s,
        "R": R.tolist(),
        "t": t.tolist(),
        "T_a_from_b_coarse": T_coarse.tolist(),
        "camera_reference": ref,
        "note": "One capture session split in two. Tests the estimator, not change over time.",
    }
    (ctx.out_dir / "T_b_to_a.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    def sub(i: dict) -> dict:
        return {k: v for k, v in i.items() if k not in ("cameras", "fused")}

    ctx.metrics = {
        "subset_a": {**sub(info_a), "n_points": int(len(pa))},
        "subset_b": {**sub(info_b), "n_points": int(len(pb))},
        "scale": s,
        "rotation_angle_deg": rotation_angle_deg(R),
        "translation": t.tolist(),
        "translation_norm": float(np.linalg.norm(t)),
        "coarse": coarse,
        "icp_converged": fine["converged"],
        "icp_iterations": fine["iterations"],
        "icp_history": fine["history"],
        "fitness": fine["fitness"],
        "inlier_rmse": fine["inlier_rmse_normalised"] * s_a,
        "units": "scene units of half A (non-metric)",
        "a_median_nn_spacing": spacing_a,
        "a_extent_diag": extent_a,
        "c2c_all": {"median": float(np.median(dist)), "p95": float(np.percentile(dist, 95))},
        "c2c_overlap": {
            "threshold": overlap_d,
            "fraction_of_b": float(ov.mean()),
            "median": float(np.median(dist[ov])),
            "p95": float(np.percentile(dist[ov], 95)),
            "signed_median": float(np.median(signed[ov])),
        },
        "camera_reference": ref,
    }
    if not fine["converged"]:
        raise AcceptanceError(f"ICP did not converge in {fine['iterations']} iterations")
