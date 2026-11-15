"""Plane fit and wall-frame axes on a synthetic tilted plane."""

from __future__ import annotations

import numpy as np

from stonetwin.stages.frame import (
    apply_T,
    fit_plane_ransac,
    make_T,
    wall_axes,
)


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = np.radians(deg)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def test_plane_and_wall_frame_on_tilted_plane() -> None:
    rng = np.random.default_rng(1)
    # Wall in its own frame: x in [-2, 2], y in [-1, 1], z ~ 0 with 0.002 noise, plus outliers.
    p_wall_true = np.c_[
        rng.uniform(-2, 2, 4000), rng.uniform(-1, 1, 4000), rng.normal(0, 0.002, 4000)
    ]
    outliers = rng.uniform(-2, 2, (600, 3))
    R_world_from_wall = _rot(np.array([0.3, 1.0, 0.2]), 35)
    o_world = np.array([5.0, -2.0, 3.0])
    p_world = np.r_[p_wall_true, outliers] @ R_world_from_wall.T + o_world

    # Cameras 3 units in front of the wall (+z_wall), OpenCV up (-y_cam) = +y_wall.
    centers = (
        np.c_[np.linspace(-2, 2, 7), np.zeros(7), np.full(7, 3.0)] @ R_world_from_wall.T + o_world
    )
    ups = np.tile(R_world_from_wall[:, 1] + 0.05 * R_world_from_wall[:, 2], (7, 1))

    n, c, inl = fit_plane_ransac(p_world, threshold=0.01, seed=0)
    assert inl[:4000].mean() > 0.99
    R = wall_axes(n, centers, ups, c)
    # Rows of R_wall_from_world must match the true wall axes in world.
    assert np.allclose(R, R_world_from_wall.T, atol=1e-2)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)

    T = make_T(R, c, scale=2.0)
    p = apply_T(T, p_world[:4000])
    assert abs(np.median(p[:, 2])) < 0.01
    assert np.allclose(p[:, 0].max() - p[:, 0].min(), 8.0, atol=0.05)  # 4 units * 2
