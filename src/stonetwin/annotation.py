"""Shapes drawn on an earlier orthophoto, mapped onto the current one (D-019).

Without metric scale each SfM run has its own gauge, so the orthophoto grid moves
between runs. Polygons drawn by hand (ground truth, exclusions) are stored in pixel
coordinates of the orthophoto they were drawn on; a copy of that image is kept as
``ortho_ref.jpg`` next to them. These helpers register the current orthophoto to the
copy and map the polygons. Pixel coordinates are (x = column, y = row).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from stonetwin.stage import StageError


def register_images(
    ref: np.ndarray, cur: np.ndarray, ratio: float = 0.75, ransac_px: float = 3.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Similarity mapping pixel coordinates of ``ref`` to ``cur`` (both RGB uint8).

    SIFT features, Lowe ratio test, RANSAC on a 4-DoF similarity
    (``cv2.estimateAffinePartial2D``).

    Returns:
        (2x3 matrix A with ``[x', y'] = A @ [x, y, 1]``, quality: n_matches, n_inliers,
        rms residual of inliers in ``cur`` pixels, scale).
    """
    sift = cv2.SIFT_create(nfeatures=20000)
    g = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in (ref, cur)]
    (k1, d1), (k2, d2) = (sift.detectAndCompute(x, None) for x in g)
    if d1 is None or d2 is None:
        raise StageError("no features found to register the annotation image")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < ratio * n.distance]
    if len(good) < 10:
        raise StageError(f"only {len(good)} feature matches between annotation and orthophoto")
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_px, maxIters=10000
    )
    if A is None:
        raise StageError("could not register the annotation image to the orthophoto")
    inl = inl.ravel().astype(bool)
    res = (src[inl] @ A[:, :2].T + A[:, 2]) - dst[inl]
    return A, {
        "n_matches": len(good),
        "n_inliers": int(inl.sum()),
        "rms_px": float(np.sqrt((res**2).sum(1).mean())),
        "scale": float(np.hypot(A[0, 0], A[1, 0])),
    }


def map_shapes(shapes: list[dict], A: np.ndarray, pre: float, post: float) -> list[dict]:
    """Map shapes: ``p_out = post * (A @ [p_in * pre, 1])``."""
    out = []
    for sh in shapes:
        q = np.asarray(sh["points"], float) * pre
        q = (q @ A[:, :2].T + A[:, 2]) * post
        out.append({**sh, "points": q.tolist()})
    return out


def shapes_on_current_grid(
    ann: dict, ref_path: Path, cur: np.ndarray
) -> tuple[list[dict], dict[str, Any]]:
    """Map a labelme document's shapes onto the pixel grid of ``cur`` (RGB uint8).

    If ``ref_path`` does not exist the shapes are returned unchanged, which is only
    valid when the annotation was drawn on an image of the same size.

    Returns:
        (shapes in ``cur`` pixels, registration info; ``n_inliers`` is absent when no
        registration was done).
    """
    if not ref_path.is_file():
        if (ann.get("imageHeight"), ann.get("imageWidth")) != cur.shape[:2]:
            raise StageError(
                "annotation image size differs from the orthophoto and there is no "
                "ortho_ref.jpg to register against"
            )
        return ann["shapes"], {"method": "none (same grid)"}
    ref = np.asarray(Image.open(ref_path).convert("RGB"))
    k = ref.shape[1] / cur.shape[1]  # register at the reference's resolution
    cur_s = cv2.resize(cur, (ref.shape[1], round(cur.shape[0] * k)), interpolation=cv2.INTER_AREA)
    A, q = register_images(ref, cur_s)
    pre = ref.shape[1] / ann["imageWidth"]
    info = {
        "method": "SIFT + RANSAC similarity to ortho_ref.jpg",
        **q,
        "rms_px_full_res": q["rms_px"] / k,
    }
    return map_shapes(ann["shapes"], A, pre, 1.0 / k), info
