"""Orthophoto rasterisation and mask resolution."""

from __future__ import annotations

import numpy as np

from stonetwin.stages.segment import apply_exclusion, rasterize, resolve_masks


def test_rasterize_keeps_front_point_and_index() -> None:
    p = np.array([[0.0, 1.0, 0.0], [0.05, 0.95, 0.5], [0.95, 0.05, 0.0], [0.02, 0.98, -0.2]])
    rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    img, relief, pix, meta = rasterize(p, rgb, gsd=0.1)
    assert (meta["width"], meta["height"]) == (10, 10)
    assert tuple(pix[2]) == (9, 9)  # bottom-right
    # Points 0, 1, 3 share pixel (0, 0); point 1 (z = 0.5) is in front.
    assert tuple(pix[0]) == tuple(pix[1]) == tuple(pix[3]) == (0, 0)
    assert tuple(img[0, 0]) == (0, 255, 0)
    assert relief[0, 0] == 0.5
    assert np.isnan(relief[5, 5])


def _box(shape: tuple[int, int], r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[r0:r1, c0:c1] = True
    return m


def test_resolve_masks_finer_level_wins_and_area_filter() -> None:
    shape = (100, 100)
    valid = np.ones(shape, bool)
    stone_a = _box(shape, 10, 30, 10, 40)  # 600 px
    stone_b = _box(shape, 10, 30, 45, 75)  # 600 px
    group = _box(shape, 5, 35, 5, 80)  # contains both stones -> dropped
    speck = _box(shape, 60, 62, 60, 62)  # 4 px -> below min area
    huge = _box(shape, 0, 100, 0, 100)  # above max area
    overlap = _box(shape, 50, 70, 10, 40)  # 600 px
    inner = _box(shape, 55, 65, 30, 50)  # 200 px, half inside `overlap`
    label, counts = resolve_masks(
        [stone_a, stone_b, group, speck, huge, overlap, inner],
        valid,
        min_area_px=50,
        max_area_px=5000,
        contain_frac=0.9,
    )
    assert counts["raw"] == 7 and counts["area_ok"] == 5
    assert counts["after_containment"] == 4  # group dropped; inner only half contained
    assert counts["stones"] == 4
    # Overlap pixels go to the smaller mask.
    assert label[60, 35] == label[60, 45] != label[60, 15]
    assert label[20, 20] != label[20, 50] and label[20, 20] > 0
    assert label[2, 2] == 0


def test_pair_counts_matches_naive_for_any_batch_size() -> None:
    from collections import Counter
    from itertools import combinations

    from stonetwin.stages.segment_multiview import pair_counts

    rng = np.random.default_rng(0)
    groups = [np.sort(rng.choice(30, rng.integers(1, 9), replace=False)) for _ in range(400)]
    naive = Counter(p for g in groups for p in combinations(g.tolist(), 2))
    nodes = np.concatenate(groups)
    starts = np.r_[0, np.cumsum([len(g) for g in groups])]
    for batch in (1, 7, 10**9):
        codes, counts = pair_counts(nodes, starts, 30, batch)
        got = {(int(c // 30), int(c % 30)): int(n) for c, n in zip(codes, counts, strict=True)}
        assert got == dict(naive)


def test_apply_exclusion_removes_mostly_inside_and_clips_the_rest() -> None:
    lab = np.zeros((10, 10), np.int32)
    lab[0:4, 0:4] = 1  # fully inside the exclusion -> removed
    lab[0:4, 6:10] = 2  # untouched -> kept, renumbered
    lab[6:10, 0:4] = 3  # one quarter inside -> kept, clipped
    inside = np.zeros((10, 10), bool)
    inside[0:4, 0:4] = True
    inside[6:8, 0:2] = True
    out, info = apply_exclusion(lab, inside, 0.5)
    assert info == {"stones_before": 3, "stones_removed": 1, "stones_clipped": 1}
    assert set(np.unique(out)) == {0, 1, 2}
    assert (out[inside] == 0).all()
    assert (out[0:4, 6:10] == 1).all()
    assert (out[8:10, 0:4] == 2).all()
    # Works on per-point labels too.
    pts, _ = apply_exclusion(np.array([1, 1, 2, 2, 0]), np.array([1, 0, 0, 0, 1], bool), 0.5)
    assert pts.tolist() == [0, 0, 1, 1, 0]
