"""Stage ``ingest``: inventory the raw photos, score blur, write the manifest.

Raw files are opened read-only and never written. The manifest keeps only the EXIF
fields needed downstream; GPS, serial numbers and owner fields are not copied.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from stonetwin.stage import StageContext, StageError

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# EXIF tag ids (Exif sub-IFD 0x8769).
_EXIF_IFD = 0x8769
_TAGS = {
    "capture_time": 0x9003,  # DateTimeOriginal
    "focal_length_mm": 0x920A,
    "iso": 0x8827,
    "exposure_s": 0x829A,
    "f_number": 0x829D,
}


def read_exif(path: Path) -> dict:
    """Return width, height and the whitelisted EXIF fields of one image."""
    with Image.open(path) as img:
        width, height = img.size
        sub = img.getexif().get_ifd(_EXIF_IFD)
    rec: dict = {"width": width, "height": height}
    for key, tag in _TAGS.items():
        value = sub.get(tag)
        if isinstance(value, tuple):  # some cameras store ISO as a tuple
            value = value[0] if value else None
        if value is not None and key != "capture_time":
            value = float(value)
        rec[key] = value
    return rec


def blur_score(path: Path) -> float:
    """Variance of the Laplacian of the grayscale image (higher = sharper)."""
    data = np.fromfile(path, dtype=np.uint8)  # handles non-ASCII paths on Windows
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise StageError(f"Could not decode {path.name}")
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def flag_blurriest(scores: np.ndarray, percentile: float) -> np.ndarray:
    """Boolean mask of the ``round(n * percentile / 100)`` lowest scores (ties by order)."""
    n_reject = int(round(len(scores) * percentile / 100.0))
    flags = np.zeros(len(scores), dtype=bool)
    if n_reject > 0:
        flags[np.argsort(scores, kind="stable")[:n_reject]] = True
    return flags


def _blur_figure(df: pd.DataFrame, out: Path, scene: str) -> None:
    d = df.sort_values("order")
    colors = np.where(d["use"], "#4C78A8", "#E45756")
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=120)
    ax.bar(d["order"], d["blur_score"], color=colors, width=0.9)
    ax.set_xlabel("capture order")
    ax.set_ylabel("variance of Laplacian")
    n_flag = int((~d["use"]).sum())
    ax.set_title(f"{scene}: blur score per image ({n_flag} flagged, red)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def run(ctx: StageContext) -> None:
    """Write ``manifest.csv`` and ``blur_report.png`` for ``ctx.scene``.

    Raises:
        StageError: no images, unreadable image, or non-constant focal length /
            resolution while ``sfm.single_camera`` is true.
    """
    icfg = ctx.cfg["ingest"]
    raw_dir = ctx.root / ctx.cfg["raw_dir"]
    files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise StageError(f"No images in {raw_dir}")
    ctx.inputs = {"raw_dir": ctx.cfg["raw_dir"], "n_files": len(files)}
    ctx.params = {
        "blur_metric": icfg["blur_metric"],
        "blur_reject_percentile": icfg["blur_reject_percentile"],
        "single_camera": ctx.cfg["sfm"]["single_camera"],
    }

    rows = []
    for path in files:
        rec = {"file": path.name, "size_bytes": path.stat().st_size, **read_exif(path)}
        rec["blur_score"] = blur_score(path)
        rows.append(rec)
    df = pd.DataFrame(rows)

    # Capture order: EXIF time, file name as tie-break (the camera clock is only
    # trusted for ordering, see docs/DATA.md).
    df = df.sort_values(["capture_time", "file"], na_position="last", kind="stable")
    df["order"] = np.arange(len(df))
    df["use"] = ~flag_blurriest(df["blur_score"].to_numpy(), icfg["blur_reject_percentile"])
    df["reason"] = np.where(df["use"], "", "blur")

    if ctx.cfg["sfm"]["single_camera"]:
        for col in ("focal_length_mm", "width", "height"):
            values = df[col].dropna().unique()
            if len(values) != 1 or df[col].isna().any():
                raise StageError(
                    f"{col} is not constant ({sorted(values)}) but sfm.single_camera is true"
                )

    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "order",
        "file",
        "use",
        "reason",
        "blur_score",
        "capture_time",
        "width",
        "height",
        "focal_length_mm",
        "iso",
        "exposure_s",
        "f_number",
        "size_bytes",
    ]
    df[cols].to_csv(ctx.out_dir / "manifest.csv", index=False)
    _blur_figure(df, ctx.out_dir / "blur_report.png", ctx.scene)

    ctx.metrics = {
        "n_images": len(df),
        "n_used": int(df["use"].sum()),
        "n_flagged_blur": int((~df["use"]).sum()),
        "flagged_files": df.loc[~df["use"], "file"].tolist(),
        "blur_score_median": float(df["blur_score"].median()),
        "blur_score_min": float(df["blur_score"].min()),
        "iso_range": [float(df["iso"].min()), float(df["iso"].max())],
        "focal_length_mm": float(df["focal_length_mm"].iloc[0]),
        "resolution": [int(df["width"].iloc[0]), int(df["height"].iloc[0])],
    }
