"""Similarity estimation: Umeyama, decomposition, split, and coarse+fine alignment."""

from __future__ import annotations

import numpy as np
import pytest

from stonetwin.stages.register import (
    apply_T,
    coarse_align,
    decompose,
    fine_align,
    normaliser,
    rotation_angle_deg,
    sim_matrix,
    split_images,
    umeyama,
)


def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = np.radians(deg)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def test_umeyama_recovers_known_similarity() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3))
    s, R, t = 2.7, _rot(np.array([1.0, -2.0, 0.5]), 73.0), np.array([4.0, -1.0, 9.0])
    dst = s * src @ R.T + t + rng.normal(scale=1e-6, size=src.shape)
    s_e, R_e, t_e = umeyama(src, dst)
    assert s_e == pytest.approx(s, rel=1e-6)
    assert rotation_angle_deg(R_e @ R.T) < 1e-4
    assert np.allclose(t_e, t, atol=1e-5)
    assert np.isclose(np.linalg.det(R_e), 1.0)


def test_umeyama_without_scale_and_reflection_guard() -> None:
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    R = _rot(np.array([0.0, 0.0, 1.0]), 30.0)
    s, R_e, _ = umeyama(src, src @ R.T, with_scale=False)
    assert s == 1.0 and rotation_angle_deg(R_e @ R.T) < 1e-6
    # A planar point set must still give a proper rotation (det +1).
    flat = np.c_[rng.normal(size=(50, 2)), np.zeros(50)]
    _, R_f, _ = umeyama(flat, flat @ R.T)
    assert np.isclose(np.linalg.det(R_f), 1.0)


def test_sim_matrix_roundtrip() -> None:
    R = _rot(np.array([1.0, 1.0, 0.0]), 40.0)
    s, R2, t = decompose(sim_matrix(0.3, R, np.array([1.0, 2.0, 3.0])))
    assert s == pytest.approx(0.3) and np.allclose(R2, R) and np.allclose(t, [1, 2, 3])


def test_split_disjoint() -> None:
    names = [f"{i:03d}" for i in range(10)]
    a, b = split_images(names, "odd_even")
    assert a == names[0::2] and b == names[1::2]
    a, b = split_images(names, "thirds")
    assert not set(a) & set(b) and len(a) == 7 and len(b) == 3


def _wall(rng: np.random.Generator, n: int) -> np.ndarray:
    """A bumpy 4 x 2 wall with a recess and a buttress, so features are distinctive."""
    x, y = rng.uniform(-2, 2, n), rng.uniform(-1, 1, n)
    z = 0.05 * np.sin(3 * x) * np.cos(4 * y) + 0.03 * np.sin(7 * x + 2 * y)
    z -= 0.3 * ((np.abs(x + 0.8) < 0.4) & (np.abs(y - 0.2) < 0.3))
    z += 0.25 * ((np.abs(x - 1.2) < 0.15) & (y < 0.5))
    return np.c_[x, y, z]


def test_coarse_and_fine_alignment_recover_similarity() -> None:
    rng = np.random.default_rng(3)
    pa = _wall(rng, 120_000)
    s, R, t = 0.37, _rot(np.array([0.2, 1.0, -0.4]), 120.0), np.array([3.0, -7.0, 1.5])
    # B is an independent sample of the same surface, in another gauge: p_a = s R p_b + t.
    pb_in_a = _wall(rng, 120_000)
    pb = (pb_in_a - t) @ R / s
    rc = {
        "voxel": 0.02,
        "normal_radius_factor": 2.5,
        "fpfh_radius_factor": 5,
        "ransac_distance_factor": 1.5,
        "ransac_max_iterations": 200_000,
        "ransac_confidence": 0.999,
        "icp_voxel": 0.01,
        "icp_max_distance_factor": 2.5,
        "icp_max_iterations": 200,
        "converge_scale": 1e-5,
        "converge_rot_deg": 1e-3,
    }
    Na, Nb = normaliser(pa), normaliser(pb)
    T0, _ = coarse_align(apply_T(Na, pa), apply_T(Nb, pb), rc, seed=0)
    Tn, info = fine_align(apply_T(Na, pa), apply_T(Nb, pb), T0, rc)
    s_e, R_e, t_e = decompose(np.linalg.inv(Na) @ Tn @ Nb)
    assert info["converged"]
    assert s_e == pytest.approx(s, rel=5e-3)
    assert rotation_angle_deg(R_e @ R.T) < 0.5
    assert np.linalg.norm(t_e - t) < 0.02
