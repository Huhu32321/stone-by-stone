"""Render both segmentation methods' point labels on the orthophoto grid, side by side.

Not a pipeline stage; for gate reports.

    python scripts/label_maps.py --scene highwall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from stonetwin.stages.segment import label_colours, overlay_boundaries


def raster_labels(p_wall: np.ndarray, labels: np.ndarray, meta: dict) -> np.ndarray:
    """Front-most point's label per orthophoto pixel, small gaps closed."""
    g = meta["gsd"]
    col = np.floor((p_wall[:, 0] - meta["x0"]) / g).astype(int)
    row = np.floor((meta["y0"] - p_wall[:, 1]) / g).astype(int)
    ok = (col >= 0) & (col < meta["width"]) & (row >= 0) & (row < meta["height"])
    o = np.argsort(p_wall[ok, 2])
    lab = np.zeros((meta["height"], meta["width"]), np.int32)
    lab[row[ok][o], col[ok][o]] = labels[ok][o]
    closed = cv2.morphologyEx(lab.astype(np.float32), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return np.where(lab == 0, closed.astype(np.int32), lab)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="highwall")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "outputs" / a.scene
    seg = root / "06_segment"
    meta = json.loads((seg / "ortho" / "ortho_meta.json").read_text(encoding="utf-8"))
    ortho = np.asarray(Image.open(seg / "ortho" / "ortho.png").convert("RGB"))
    p = np.asarray(o3d.io.read_point_cloud(str(root / "04_frame" / "dense_wall.ply")).points)
    panels = []
    for m in ("ortho", "multiview"):
        f = seg / m / "points_labels.npy"
        if not f.is_file():
            continue
        lab = raster_labels(p, np.load(f), meta)
        Image.fromarray(label_colours(lab)).save(seg / m / "label_map.png")
        ov = overlay_boundaries(ortho, lab)
        Image.fromarray(ov).save(seg / m / "label_map_overlay.png")
        panels.append(ov)
    gap = np.full((panels[0].shape[0], 20, 3), 255, np.uint8)
    both = Image.fromarray(np.concatenate([panels[0], gap, panels[1]], axis=1))
    both.thumbnail((3000, 3000))
    both.save(seg / "methods_side_by_side.jpg", quality=88)
    print(seg / "methods_side_by_side.jpg")


if __name__ == "__main__":
    main()
