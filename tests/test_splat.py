"""Held-out comparison image splitting and PSNR."""

from __future__ import annotations

import numpy as np

from stonetwin.stages.splat import psnr, split_comparison


def test_split_comparison() -> None:
    gt = np.full((5, 7, 3), 10, np.uint8)
    rd = np.full((5, 7, 3), 200, np.uint8)
    img = np.concatenate([gt, np.zeros((5, 4, 3), np.uint8), rd], axis=1)
    a, b = split_comparison(img)
    assert (a == gt).all() and (b == rd).all()


def test_psnr() -> None:
    a = np.zeros((4, 4, 3))
    b = np.full((4, 4, 3), 0.1)
    assert np.isclose(psnr(a, b), 20.0)
