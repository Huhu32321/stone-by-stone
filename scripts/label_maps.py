"""Render both segmentation methods' point labels on the orthophoto grid, side by side.

Not a pipeline stage; for gate reports.

    python scripts/label_maps.py --scene highwall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from stonetwin.stages.segment import (
    label_colours,
    overlay_boundaries,
)
from stonetwin.stages.segment import (
    points_to_ortho_labels as raster_labels,
)


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
