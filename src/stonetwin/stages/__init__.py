"""Pipeline stages. Each module exposes ``run(ctx: StageContext) -> None``."""

from __future__ import annotations

from stonetwin.stage import StageFn
from stonetwin.stages import (
    attributes,
    dense,
    evaluate,
    export_viewer,
    frame,
    ingest,
    register,
    register_control,
    report,
    segment,
    sfm,
    splat,
)

STAGES: dict[str, StageFn] = {
    "ingest": ingest.run,
    "sfm": sfm.run,
    "dense": dense.run,
    "splat": splat.run,
    "frame": frame.run,
    "register": register.run,
    "register-control": register_control.run,
    "segment": segment.run,
    "attributes": attributes.run,
    "evaluate": evaluate.run,
    "export-viewer": export_viewer.run,
    "report": report.run,
}

HELP: dict[str, str] = {
    "ingest": "Read photos, extract EXIF, score blur, write the manifest.",
    "sfm": "COLMAP feature extraction, matching and mapping; export cameras.json.",
    "dense": "Dense point cloud (COLMAP MVS or splat depth fusion).",
    "splat": "Train 3D Gaussian Splatting with a held-out test split.",
    "frame": "Fit the wall plane, define the wall frame, apply scale, crop.",
    "register": "Independent half-reconstructions aligned by a similarity transform.",
    "register-control": "Control for register: the same halves with intrinsics held fixed.",
    "segment": "Segment stones (ortho baseline and multi-view lifting).",
    "attributes": "Per-stone geometry and colour attributes (condition proxies).",
    "evaluate": "Compare segmentation with hand-annotated ground truth.",
    "export-viewer": "Write the downsampled viewer payload.",
    "report": "Merge metrics and regenerate RESULTS.md, figures and README block.",
}
