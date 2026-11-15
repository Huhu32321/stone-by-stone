"""Attributes on a synthetic stone of known size, roughness, recession and colour."""

from __future__ import annotations

import numpy as np
import pytest

from stonetwin.stages.attributes import FIELDS, stone_records, validate


def _grid(x0: float, x1: float, y0: float, y1: float, step: float) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.c_[xs.ravel(), ys.ravel()]


def test_synthetic_stone() -> None:
    step = 0.004
    # Target stone: 0.40 wide x 0.20 high, set back by 0.05, with a known ripple.
    xy = _grid(0.0, 0.40, 0.0, 0.20, step)
    amp = 0.004
    z = -0.05 + amp * np.sin(2 * np.pi * xy[:, 0] / 0.1)
    stone = np.c_[xy, z]
    pts, lab, col = [stone], [np.full(len(stone), 1)], [np.tile([0.6, 0.3, 0.2], (len(stone), 1))]
    # Six neighbours on a ring, flush with the wall (z = 0), grey.
    for k, (dx, dy) in enumerate(
        [(-0.5, 0), (0.5, 0), (0, 0.3), (0, -0.3), (-0.5, 0.3), (0.5, -0.3)], start=2
    ):
        g = _grid(dx, dx + 0.3, dy, dy + 0.15, step)
        pts.append(np.c_[g, np.zeros(len(g))])
        lab.append(np.full(len(g), k))
        col.append(np.full((len(g), 3), 0.5))
    p, labels, rgb = np.concatenate(pts), np.concatenate(lab), np.concatenate(col)

    recs = stone_records(
        p, rgb, labels, gsd=step, close_px=5, k_neighbours=6, min_points=10, radius=0.4
    )
    assert validate(recs) == []
    r = next(r for r in recs if r["id"] == 1)
    assert set(FIELDS) <= set(r)
    assert r["face_area"] == pytest.approx(0.40 * 0.20, rel=0.05)
    assert r["width"] == pytest.approx(0.40, rel=0.03)
    assert r["height"] == pytest.approx(0.20, rel=0.05)
    assert r["aspect_ratio"] == pytest.approx(2.0, rel=0.06)
    assert r["roughness_rms"] == pytest.approx(amp / np.sqrt(2), rel=0.05)
    assert r["recession"] == pytest.approx(-0.05, abs=1e-3)
    assert r["centroid_wall"][0] == pytest.approx(0.2, abs=step)
    # Reddish stone on a grey wall: clearly different colour, a* > 0.
    assert r["colour_lab"][1] > 10 and r["colour_delta_e"] > 10
    grey = next(r for r in recs if r["id"] == 2)
    assert grey["roughness_rms"] < 1e-9 and grey["colour_std"] < 1e-6
