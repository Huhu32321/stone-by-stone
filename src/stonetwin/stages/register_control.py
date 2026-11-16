"""Stage ``register-control``: the registration experiment with intrinsics held fixed.

Control for the ``register`` stage. The same two image subsets are reconstructed
again, but every SfM run starts from the intrinsics (focal length, principal point,
distortion) of the full-scene model in ``01_sfm`` and bundle adjustment keeps them
fixed. Alignment, camera-pose reference and residual measures are identical to
``register``.

Question it answers: does the smooth bowl in the half-to-half residual come from each
half calibrating its own lens? If it does, fixing the calibration should shrink the
bowl and the scale error. The stage reports both runs side by side; it draws no
conclusion itself.

Lengths are in the scene units of each run's half A (non-metric); ``*_rel_extent``
values are comparable across runs.
"""

from __future__ import annotations

import json

import numpy as np
import open3d as o3d

from stonetwin.stage import AcceptanceError, StageContext, StageError
from stonetwin.stages.register import (
    align_and_measure,
    doming,
    reconstruct_subset,
    signed_residual,
)
from stonetwin.tools import colmap_version, find_tool


def _baseline(ctx: StageContext) -> dict:
    """Residual measures of the existing ``register`` run, from its saved clouds."""
    rd = ctx.stage_dir("register")
    rec = json.loads((rd / "stage.json").read_text(encoding="utf-8"))
    m = rec["metrics"]
    pa = np.asarray(o3d.io.read_point_cloud(str(rd / "subset_a.ply")).points)
    pb_in_a = np.asarray(o3d.io.read_point_cloud(str(rd / "subset_b_in_a.ply")).points)
    dist, signed = signed_residual(pa, pb_in_a)
    ov = dist < m["c2c_overlap"]["threshold"]
    ref = m["camera_reference"]
    return {
        "scale_rel_error": ref["scale_rel_error"],
        "rotation_error_deg": ref["rotation_error_deg"],
        "translation_error_rel_extent": ref["translation_error_rel_extent"],
        "c2c_overlap": m["c2c_overlap"],
        "c2c_overlap_median_rel_extent": m["c2c_overlap"]["median"] / m["a_extent_diag"],
        "doming": doming(pb_in_a[ov], signed[ov], m["a_extent_diag"]),
        "camera_params": {
            "a": m["subset_a"].get("camera_params"),
            "b": m["subset_b"].get("camera_params"),
        },
        "mean_reproj_error_px": {
            "a": m["subset_a"]["mean_reproj_error_px"],
            "b": m["subset_b"]["mean_reproj_error_px"],
        },
    }


def run(ctx: StageContext) -> None:
    """Reconstruct both halves with fixed full-scene intrinsics, align, compare.

    Raises:
        StageError: ``register`` or ``sfm`` has not run, or the model has several cameras.
        AcceptanceError: ICP does not converge on the control halves.
    """
    rc = dict(ctx.cfg["register"])
    seed = int(ctx.cfg["seed"])
    colmap = find_tool("colmap", ctx.cfg)
    ctx.tools = {"colmap": colmap_version(colmap), "open3d": o3d.__version__}
    rd = ctx.stage_dir("register")
    if not (rd / "stage.json").is_file():
        raise StageError("run `register` first; the control reuses its image split")
    reg_inputs = json.loads((rd / "stage.json").read_text(encoding="utf-8"))["inputs"]
    full = json.loads((ctx.stage_dir("sfm") / "cameras.json").read_text(encoding="utf-8"))
    if len(full["cameras"]) != 1:
        raise StageError("fixed-intrinsics control expects a single camera in 01_sfm")
    cam = next(iter(full["cameras"].values()))
    params = [float(v) for v in cam["params"]]
    ctx.inputs = {
        "split_from": "05_register/stage.json",
        "intrinsics_from": "01_sfm/cameras.json",
        "images_a": reg_inputs["images_a"],
        "images_b": reg_inputs["images_b"],
    }
    ctx.params = {**rc, "fixed_camera_model": cam["model"], "fixed_camera_params": params}

    info_a = reconstruct_subset(ctx, "a", reg_inputs["images_a"], colmap, params)
    info_b = reconstruct_subset(ctx, "b", reg_inputs["images_b"], colmap, params)
    out, m, fine = align_and_measure(ctx, info_a, info_b, full, rc, seed)
    (ctx.out_dir / "T_b_to_a.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    ref = m["camera_reference"]
    ctx.metrics = {
        "baseline": _baseline(ctx),
        "fixed_intrinsics": {
            **m,
            "scale_rel_error": ref["scale_rel_error"],
            "rotation_error_deg": ref["rotation_error_deg"],
            "translation_error_rel_extent": ref["translation_error_rel_extent"],
            "c2c_overlap_median_rel_extent": m["c2c_overlap"]["median"] / m["a_extent_diag"],
        },
        "note": "baseline = register stage (each half refines its own intrinsics); "
        "fixed_intrinsics = same halves, intrinsics fixed to the full-scene model.",
    }
    if not fine["converged"]:
        raise AcceptanceError(f"ICP did not converge in {fine['iterations']} iterations")
