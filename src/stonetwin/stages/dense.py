"""Stage ``dense``: dense point cloud from multi-view stereo.

Backend ``colmap_mvs``: COLMAP ``image_undistorter`` -> ``patch_match_stereo`` (with
geometric consistency) -> ``stereo_fusion``. The undistorted workspace is kept in
``02_dense/mvs/`` because the ``multiview`` segmentation method reuses its depth maps
(``mvs/stereo/depth_maps/<image>.geometric.bin``) for the visibility test.

Frames: ``dense.ply`` is in the SfM ``world`` frame (arbitrary origin, orientation and
scale; non-metric). The ``frame`` stage maps it into the wall frame.

Backend ``splat_depth_fusion`` (spec fallback when COLMAP MVS cannot run) is not
implemented; see ``docs/DECISIONS.md`` D-012.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from stonetwin.stage import AcceptanceError, StageContext, StageError
from stonetwin.tools import colmap_version, find_tool, run_logged


def median_nn_spacing(points: np.ndarray, sample: int = 200_000, seed: int = 0) -> float:
    """Median distance from a point to its nearest neighbour, in the cloud's units.

    Computed on a random subsample of at most ``sample`` query points against the full
    cloud.
    """
    if len(points) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=min(sample, len(points)), replace=False)
    dist, _ = cKDTree(points).query(points[idx], k=2)
    return float(np.median(dist[:, 1]))


def _sfm_model(ctx: StageContext) -> Path:
    import json

    sfm_json = ctx.stage_dir("sfm") / "stage.json"
    if not sfm_json.is_file():
        raise StageError(f"{sfm_json} missing; run `sfm` first")
    model_rel = json.loads(sfm_json.read_text(encoding="utf-8"))["metrics"]["model"]
    return ctx.stage_dir("sfm") / model_rel


def _colmap_mvs(ctx: StageContext, dcfg: dict) -> Path:
    colmap = find_tool("colmap", ctx.cfg)
    ctx.tools = {"colmap": colmap_version(colmap)}
    raw_dir = ctx.root / ctx.cfg["raw_dir"]
    model = _sfm_model(ctx)
    ws = ctx.out_dir / "mvs"
    if ws.exists():  # stale run; start clean, inside out_dir only
        shutil.rmtree(ws)
    log = ctx.out_dir / "colmap.log"
    log.unlink(missing_ok=True)
    seed = str(ctx.cfg["seed"])

    run_logged(
        [
            colmap,
            "image_undistorter",
            "--image_path",
            raw_dir,
            "--input_path",
            model,
            "--output_path",
            ws,
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(dcfg["max_image_size"]),
        ],
        log,
    )
    run_logged(
        [
            colmap,
            "patch_match_stereo",
            "--workspace_path",
            ws,
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
            "--PatchMatchStereo.max_image_size",
            str(dcfg["max_image_size"]),
            "--default_random_seed",
            seed,
        ],
        log,
    )
    n_depth = len(list((ws / "stereo" / "depth_maps").glob("*.geometric.bin")))
    if n_depth == 0:
        raise StageError(f"patch_match_stereo wrote no geometric depth maps; see {log}")
    fused = ws / "fused.ply"
    run_logged(
        [
            colmap,
            "stereo_fusion",
            "--workspace_path",
            ws,
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            fused,
            "--StereoFusion.min_num_pixels",
            str(dcfg["fusion_min_num_pixels"]),
            "--StereoFusion.max_reproj_error",
            str(dcfg["fusion_max_reproj_error_px"]),
            "--StereoFusion.max_depth_error",
            str(dcfg["fusion_max_depth_error"]),
            "--StereoFusion.max_normal_error",
            str(dcfg["fusion_max_normal_error_deg"]),
        ],
        log,
    )
    ctx.metrics["n_depth_maps"] = n_depth
    return fused


def run(ctx: StageContext) -> None:
    """Build ``dense.ply`` (positions, normals, RGB; SfM world frame, scene units).

    Raises:
        StageError: backend unavailable or a COLMAP step failed.
        AcceptanceError: the fused cloud is empty.
    """
    dcfg = dict(ctx.cfg["dense"])
    backend = dcfg["backend"]
    if backend == "auto":
        backend = "colmap_mvs"
    ctx.params = {**dcfg, "backend_resolved": backend}
    ctx.inputs = {"sfm_model": _sfm_model(ctx).relative_to(ctx.scene_dir).as_posix()}

    if backend == "colmap_mvs":
        fused = _colmap_mvs(ctx, dcfg)
    elif backend == "splat_depth_fusion":
        raise StageError(
            "dense.backend splat_depth_fusion is not implemented (see DECISIONS D-012); "
            "use colmap_mvs"
        )
    else:
        raise StageError(f"unknown dense.backend '{backend}'")

    pcd = o3d.io.read_point_cloud(str(fused))
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise AcceptanceError(f"fused cloud {fused} is empty")
    if not pcd.has_normals():
        pcd.estimate_normals()
    out = ctx.out_dir / "dense.ply"
    o3d.io.write_point_cloud(str(out), pcd, write_ascii=False, compressed=False)
    ctx.metrics.update(
        {
            "n_points": int(len(pts)),
            "median_nn_spacing": median_nn_spacing(pts, seed=int(ctx.cfg["seed"])),
            "units": "scene (non-metric, SfM world frame)",
            "has_colors": bool(pcd.has_colors()),
        }
    )
