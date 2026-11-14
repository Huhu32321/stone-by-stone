"""Stage ``sfm``: COLMAP feature extraction, matching and incremental mapping.

COLMAP reads the photos in place from ``raw_dir`` (read-only) through an image list
that holds only the images the manifest marks ``use = true``.

Frames: COLMAP's model frame is called ``world`` here. It is arbitrary in origin,
orientation and scale (non-metric) until the ``frame`` stage fixes it.
``cameras.json`` stores ``T_world_from_cam`` (4x4, camera-to-world), so a point
``p_cam`` in the OpenCV camera frame (x right, y down, z forward) maps to
``p_world = T_world_from_cam @ [p_cam, 1]``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap

from stonetwin.stage import AcceptanceError, StageContext, StageError
from stonetwin.tools import colmap_version, find_tool, run_logged


def _read_manifest(ctx: StageContext) -> pd.DataFrame:
    path = ctx.stage_dir("ingest") / "manifest.csv"
    if not path.is_file():
        raise StageError(f"{path} missing; run `ingest` first")
    return pd.read_csv(path)


def largest_model(sparse_dir: Path) -> tuple[Path, pycolmap.Reconstruction, int]:
    """Return (path, reconstruction, number of models) for the model with most images."""
    models = sorted(p for p in sparse_dir.iterdir() if p.is_dir())
    if not models:
        raise AcceptanceError(f"COLMAP mapper produced no model in {sparse_dir}")
    recs = [(p, pycolmap.Reconstruction(p)) for p in models]
    path, rec = max(recs, key=lambda pr: pr[1].num_reg_images())
    return path, rec, len(models)


def export_cameras(rec: pycolmap.Reconstruction, model_rel: str) -> dict:
    """Convert a COLMAP reconstruction to the ``cameras.json`` dict.

    Intrinsics are in pixels of the images COLMAP saw. For the OPENCV model,
    ``params = [fx, fy, cx, cy, k1, k2, p1, p2]``.
    """
    cameras = {}
    for cam_id, cam in rec.cameras.items():
        params = [float(v) for v in cam.params]
        cameras[str(cam_id)] = {
            "model": cam.model.name,
            "width": int(cam.width),
            "height": int(cam.height),
            "params": params,
            "K": cam.calibration_matrix().tolist(),
            "dist": params[4:] if cam.model.name == "OPENCV" else [],
        }
    images = []
    for img in sorted(rec.images.values(), key=lambda im: im.name):
        if not img.has_pose:
            continue
        T_cam_from_world = np.eye(4)
        T_cam_from_world[:3, :] = img.cam_from_world().matrix()
        T_world_from_cam = np.linalg.inv(T_cam_from_world)
        images.append(
            {
                "name": img.name,
                "image_id": int(img.image_id),
                "camera_id": int(img.camera_id),
                "T_world_from_cam": T_world_from_cam.tolist(),
                "center_world": T_world_from_cam[:3, 3].tolist(),
            }
        )
    return {
        "frame": "colmap_world",
        "metric": False,
        "convention": "OpenCV camera (x right, y down, z forward); T_world_from_cam 4x4",
        "sparse_model": model_rel,
        "cameras": cameras,
        "images": images,
    }


def colmap_sfm(
    colmap: Path,
    image_dir: Path,
    names: list[str],
    work: Path,
    scfg: dict,
    seed: int,
    log: Path,
) -> Path:
    """Run COLMAP extraction, matching and mapping on ``names`` read from ``image_dir``.

    ``work`` is deleted and recreated (callers pass a folder inside their own stage
    output). Images are read in place, never copied or modified.

    Returns:
        The ``sparse`` folder holding one sub-folder per model.
    """
    if work.exists():
        shutil.rmtree(work)
    sparse = work / "sparse"
    sparse.mkdir(parents=True)
    db = work / "database.db"
    image_list = work / "image_list.txt"
    image_list.write_text("\n".join(names) + "\n", encoding="utf-8")
    log.unlink(missing_ok=True)
    seed_s = str(seed)
    matcher = scfg["matcher"]
    if matcher not in ("exhaustive", "sequential"):
        raise StageError(f"sfm.matcher must be exhaustive or sequential, got {matcher}")
    run_logged(
        [
            colmap,
            "feature_extractor",
            "--database_path",
            db,
            "--image_path",
            image_dir,
            "--image_list_path",
            image_list,
            "--ImageReader.camera_model",
            scfg["camera_model"],
            "--ImageReader.single_camera",
            str(int(scfg["single_camera"])),
            "--SiftExtraction.max_num_features",
            str(scfg["max_num_features"]),
            "--default_random_seed",
            seed_s,
        ],
        log,
    )
    run_logged(
        [
            colmap,
            f"{matcher}_matcher",
            "--database_path",
            db,
            "--TwoViewGeometry.random_seed",
            seed_s,
            "--default_random_seed",
            seed_s,
        ],
        log,
    )
    run_logged(
        [
            colmap,
            "mapper",
            "--database_path",
            db,
            "--image_path",
            image_dir,
            "--output_path",
            sparse,
            "--Mapper.random_seed",
            seed_s,
            "--default_random_seed",
            seed_s,
        ],
        log,
    )
    return sparse


def run(ctx: StageContext) -> None:
    """Run COLMAP SfM on the usable images and export ``cameras.json``.

    Raises:
        AcceptanceError: registered fraction or mean reprojection error fails
            ``sfm.accept``. ``stage.json`` is still written with the metrics.
    """
    scfg = ctx.cfg["sfm"]
    colmap = find_tool("colmap", ctx.cfg)
    raw_dir = ctx.root / ctx.cfg["raw_dir"]
    manifest = _read_manifest(ctx)
    used = manifest.loc[manifest["use"].astype(bool), "file"].tolist()
    if len(used) < 3:
        raise StageError(f"Only {len(used)} usable images in the manifest")

    work = ctx.out_dir / "colmap"
    log = ctx.out_dir / "colmap.log"
    matcher = scfg["matcher"]
    ctx.inputs = {"manifest": "00_ingest/manifest.csv", "n_images_used": len(used)}
    ctx.params = {
        "camera_model": scfg["camera_model"],
        "single_camera": scfg["single_camera"],
        "matcher": matcher,
        "max_num_features": scfg["max_num_features"],
        "seed": ctx.cfg["seed"],
        "accept": scfg["accept"],
    }
    ctx.tools = {"colmap": colmap_version(colmap), "pycolmap": pycolmap.__version__}
    sparse = colmap_sfm(colmap, raw_dir, used, work, scfg, int(ctx.cfg["seed"]), log)

    model_dir, rec, n_models = largest_model(sparse)
    model_rel = model_dir.relative_to(ctx.out_dir).as_posix()
    cams = export_cameras(rec, model_rel)
    (ctx.out_dir / "cameras.json").write_text(json.dumps(cams, indent=1), encoding="utf-8")

    registered = {im["name"] for im in cams["images"]}
    n_reg = len(registered)
    ctx.metrics = {
        "n_images_input": len(used),
        "n_images_registered": n_reg,
        "registered_fraction": n_reg / len(used),
        "registered_fraction_of_all": n_reg / len(manifest),
        "mean_reproj_error_px": float(rec.compute_mean_reprojection_error()),
        "n_points3d": int(rec.num_points3D()),
        "mean_track_length": float(rec.compute_mean_track_length()),
        "n_models": n_models,
        "model": model_rel,
        "unregistered": sorted(set(used) - registered),
    }

    acc = scfg["accept"]
    failures = []
    if ctx.metrics["registered_fraction"] < acc["min_registered_fraction"]:
        failures.append(
            f"registered fraction {ctx.metrics['registered_fraction']:.3f} "
            f"< {acc['min_registered_fraction']}"
        )
    if ctx.metrics["mean_reproj_error_px"] > acc["max_mean_reproj_error_px"]:
        failures.append(
            f"mean reprojection error {ctx.metrics['mean_reproj_error_px']:.3f} px "
            f"> {acc['max_mean_reproj_error_px']}"
        )
    if failures:
        raise AcceptanceError("; ".join(failures))
