"""Stage ``export-viewer``: compact payload for the static web viewer (TECH_SPEC 4.9).

Writes into ``viewer/public/data/<scene>/``:

``points.bin``  little-endian, ``N`` records laid out as three planes:
    positions ``uint16[N, 3]`` quantised over the bounding box in ``meta.json``,
    colours ``uint8[N, 3]`` (sRGB), stone IDs ``uint16[N]`` (0 = unassigned).
``meta.json``   point count, bounding box, units, metric flag, frame, method.
``stones.json`` per-stone records from ``07_attributes`` (outline simplified).

Also refreshes ``viewer/public/data/scenes.json``. Positions are in the wall frame
(x along the wall, y up, z out of the wall), which three.js renders y-up as is.
No images are copied, so there is no EXIF to strip.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from stonetwin.stage import AcceptanceError, StageContext, StageError
from stonetwin.stages.attributes import PROXY_NOTE
from stonetwin.stages.segment import load_wall


def quantise(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map points to uint16 over their bounding box. Returns (q, lo, hi)."""
    lo, hi = p.min(0), p.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = np.round((p - lo) / span * 65535).astype(np.uint16)
    return q, lo, hi


def simplify(outline: list, tol: float) -> list:
    """Douglas-Peucker simplification of a closed outline (wall units)."""
    if len(outline) < 4:
        return outline
    a = np.asarray(outline, np.float32)[:, None, :]
    return cv2.approxPolyDP(a, tol, True)[:, 0, :].round(5).tolist()


def run(ctx: StageContext) -> None:
    """Write the viewer payload for ``ctx.scene``.

    Raises:
        AcceptanceError: payload larger than ``viewer.max_payload_mb``.
    """
    vcfg = dict(ctx.cfg["viewer"])
    ctx.params = vcfg
    seed = int(ctx.cfg["seed"])
    attr = ctx.stage_dir("attributes") / "stones.json"
    if not attr.is_file():
        raise StageError(f"{attr} missing; run `attributes` first")
    doc = json.loads(attr.read_text(encoding="utf-8"))
    method = doc["method"]
    labels = np.load(ctx.stage_dir("segment") / method / "points_labels.npy")
    p, rgb, tf = load_wall(ctx)
    ctx.inputs = {
        "cloud": "04_frame/dense_wall.ply",
        "stones": "07_attributes/stones.json",
        "labels": f"06_segment/{method}/points_labels.npy",
    }

    n = len(p)
    if n > vcfg["max_points"]:
        idx = np.sort(np.random.default_rng(seed).choice(n, vcfg["max_points"], replace=False))
    else:
        idx = np.arange(n)
    known = {s["id"] for s in doc["stones"]}
    lab = labels[idx]
    lab[~np.isin(lab, list(known))] = 0  # stones skipped by `attributes` show as unassigned
    if lab.max() > 65535:
        raise StageError("more than 65535 stones; uint16 IDs overflow")
    q, lo, hi = quantise(p[idx])
    col = np.clip(rgb[idx] * 255 + 0.5, 0, 255).astype(np.uint8)

    out = ctx.root / "viewer" / "public" / "data" / ctx.scene
    out.mkdir(parents=True, exist_ok=True)
    with (out / "points.bin").open("wb") as f:
        f.write(q.astype("<u2").tobytes())
        f.write(col.tobytes())
        f.write(lab.astype("<u2").tobytes())

    tol = vcfg["outline_tolerance_px"] * (doc_gsd := _gsd(ctx, tf))
    stones = []
    for s in doc["stones"]:
        s = dict(s)
        s["outline_wall"] = simplify(s["outline_wall"], tol)
        s.pop("neighbours", None)
        stones.append(s)
    meta = {
        "scene": ctx.scene,
        "n_points": int(len(idx)),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "units": tf["units"],
        "metric": tf["metric"],
        "frame": doc["frame"],
        "method": method,
        "layout": "u16 xyz | u8 rgb | u16 id",
        "outline_gsd": doc_gsd,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    sdoc = {k: v for k, v in doc.items() if k != "stones"}
    sdoc["note"] = PROXY_NOTE
    sdoc["stones"] = stones
    (out / "stones.json").write_text(json.dumps(sdoc, separators=(",", ":")), encoding="utf-8")

    scenes_path = out.parent / "scenes.json"
    scenes = json.loads(scenes_path.read_text(encoding="utf-8")) if scenes_path.is_file() else []
    scenes = sorted({*scenes, ctx.scene})
    scenes_path.write_text(json.dumps(scenes), encoding="utf-8")

    size_mb = sum(f.stat().st_size for f in out.iterdir()) / 2**20
    ctx.metrics = {
        "n_points": int(len(idx)),
        "n_stones": len(stones),
        "payload_mb": round(size_mb, 2),
        "max_payload_mb": vcfg["max_payload_mb"],
        "out_dir": out.relative_to(ctx.root).as_posix(),
    }
    if size_mb > vcfg["max_payload_mb"]:
        raise AcceptanceError(f"payload {size_mb:.1f} MB > {vcfg['max_payload_mb']} MB")


def _gsd(ctx: StageContext, tf: dict) -> float:
    from stonetwin.stages.segment import scale_params

    return scale_params(ctx.cfg["segment"], tf["metric"])["gsd"]
