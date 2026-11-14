"""Segmentation method ``multiview``: lift per-image SAM 2 masks onto the 3D points.

Images, intrinsics and poses come from the undistorted MVS workspace
(``02_dense/mvs``: PINHOLE cameras, ``sparse/`` model in the SfM ``world`` frame), and
visibility from its geometric depth maps. Points of ``dense_wall.ply`` (wall frame) are
mapped back to ``world`` with ``inv(T_wall_from_world)`` before projection.

Steps (TECH_SPEC 4.6): masks per image -> per-image point-to-mask assignment with a depth
test -> graph of masks linked across images by point-set IoU -> connected components =
stones -> per-point majority vote -> area filter in the wall plane.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from stonetwin.stage import StageContext, StageError
from stonetwin.stages.segment import label_colours, resolve_masks, scale_params


def read_colmap_array(path: Path) -> np.ndarray:
    """Read a COLMAP dense ``.bin`` array (depth or normal map) as (H, W[, C])."""
    with path.open("rb") as f:
        header = b""
        amp = 0
        while amp < 3:
            ch = f.read(1)
            header += ch
            amp += ch == b"&"
        w, h, c = (int(x) for x in header.decode().strip("&").split("&"))
        data = np.fromfile(f, np.float32)
    arr = data.reshape((w, h, c), order="F").transpose(1, 0, 2)
    return arr[:, :, 0] if c == 1 else arr


def project(
    p_world: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinhole projection. Returns (col, row, depth); col/row are floor pixel indices.

    COLMAP convention: pixel (0, 0) covers [0, 1) x [0, 1).
    """
    pc = p_world @ R.T + t
    z = pc[:, 2]
    zs = np.where(z > 1e-9, z, np.nan)
    u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
    return np.floor(u), np.floor(v), z


def visible_pixels(
    p_world: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
    rel_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices of points that pass the depth test, with their (row, col).

    A point is visible if it projects inside the image, in front of the camera, and its
    depth agrees with the depth map to ``rel_tol`` (relative). Depth-map pixels with
    value <= 0 (no estimate) reject the point.
    """
    H, W = depth.shape
    col, row, z = project(p_world, R, t, K)
    ok = np.isfinite(col) & (col >= 0) & (col < W) & (row >= 0) & (row < H) & (z > 0)
    idx = np.nonzero(ok)[0]
    r, c = row[idx].astype(int), col[idx].astype(int)
    d = depth[r, c]
    good = (d > 0) & (np.abs(z[idx] - d) <= rel_tol * d)
    return idx[good], r[good], c[good]


def _views(mvs: Path) -> list[dict[str, Any]]:
    rec = pycolmap.Reconstruction(mvs / "sparse")
    views = []
    for img in sorted(rec.images.values(), key=lambda im: im.name):
        cam = rec.cameras[img.camera_id]
        M = img.cam_from_world().matrix()
        views.append(
            {
                "name": img.name,
                "R": M[:, :3],
                "t": M[:, 3],
                "K": cam.calibration_matrix(),
                "w": cam.width,
                "h": cam.height,
            }
        )
    return views


def _overlay(
    img: np.ndarray, label_img: np.ndarray, pts_rc: tuple, pts_lab: np.ndarray, seed: int
) -> np.ndarray:
    """Left: SAM masks of the view. Right: final stone labels of visible points."""
    left = (0.5 * img + 0.5 * label_colours(label_img, seed + 1)).astype(np.uint8)
    right = (img * 0.35).astype(np.uint8)
    pal = label_colours(np.arange(pts_lab.max() + 1)[None], seed)[0]
    r, c = pts_rc
    has = pts_lab > 0
    for dr in (0, 1):
        for dc in (0, 1):
            rr = np.clip(r[has] + dr, 0, img.shape[0] - 1)
            cc = np.clip(c[has] + dc, 0, img.shape[1] - 1)
            right[rr, cc] = pal[pts_lab[has]]
    return np.concatenate([left, np.full((img.shape[0], 8, 3), 255, np.uint8), right], 1)


def run_multiview(ctx: StageContext, p_wall: np.ndarray, rgb: np.ndarray, tf: dict) -> dict:
    """Multi-view method. Writes ``06_segment/multiview/``; returns metrics."""
    from stonetwin.sam import generate, load_generator

    seg = ctx.cfg["segment"]
    mv = seg["multiview"]
    seed = int(ctx.cfg["seed"])
    out = ctx.out_dir / "multiview"
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(parents=True, exist_ok=True)
    mvs = ctx.stage_dir("dense") / "mvs"
    if not (mvs / "sparse").is_dir():
        raise StageError(f"{mvs} missing; multiview needs the colmap_mvs workspace")

    T = np.array(tf["T_wall_from_world"])
    p_world = (p_wall - T[:3, 3]) @ np.linalg.inv(T[:3, :3]).T
    views = _views(mvs)
    n_pts = len(p_wall)

    # 1-2. Masks per image, and point -> mask (global node id) per image.
    cfg_sam = dict(ctx.cfg)
    cfg_sam["segment"] = {**seg, "sam": {**seg["sam"], **mv["sam"]}}
    gen = load_generator(cfg_sam)
    obs_pt, obs_node, obs_view = [], [], []
    node_view: list[int] = []
    t_sam = 0.0
    per_view = []
    for vi, v in enumerate(views):
        img = np.asarray(Image.open(mvs / "images" / v["name"]).convert("RGB"))
        depth = read_colmap_array(mvs / "stereo" / "depth_maps" / f"{v['name']}.geometric.bin")
        if depth.shape != img.shape[:2]:
            raise StageError(f"depth map and image size differ for {v['name']}")
        t0 = time.perf_counter()
        recs = generate(gen, img)
        t_sam += time.perf_counter() - t0
        area_img = img.shape[0] * img.shape[1]
        lab, _ = resolve_masks(
            [r["segmentation"] for r in recs],
            np.ones(img.shape[:2], bool),
            mv["min_mask_px"],
            mv["max_mask_frac"] * area_img,
            seg["contain_frac"],
        )
        Image.fromarray(lab.astype(np.uint16)).save(out / "masks" / f"{Path(v['name']).stem}.png")
        idx, r, c = visible_pixels(p_world, v["R"], v["t"], v["K"], depth, mv["depth_rel_tol"])
        m = lab[r, c]
        keep = m > 0
        base = len(node_view)
        node_view.extend([vi] * int(lab.max()))
        obs_pt.append(idx[keep])
        obs_node.append(base + m[keep] - 1)
        obs_view.append(np.full(int(keep.sum()), vi, np.int32))
        per_view.append(
            {
                "image": v["name"],
                "n_masks": int(lab.max()),
                "n_visible_points": int(len(idx)),
                "n_points_in_masks": int(keep.sum()),
                "_lab": lab if vi in _debug_views(len(views), mv["n_debug_views"]) else None,
                "_rc": (r, c),
                "_idx": idx,
            }
        )
    obs_pt = np.concatenate(obs_pt)
    obs_node = np.concatenate(obs_node)
    obs_view = np.concatenate(obs_view)
    n_nodes = len(node_view)
    node_view_arr = np.array(node_view)

    # 3. Mask graph: IoU of the point sets of two masks from different images, on a
    # random subset of points (graph_point_fraction) to bound the pair count.
    rng = np.random.default_rng(seed)
    sampled = rng.random(n_pts) < mv["graph_point_fraction"]
    so = sampled[obs_pt]
    sp, sn = obs_pt[so], obs_node[so]
    order = np.argsort(sp, kind="stable")
    sp, sn = sp[order], sn[order]
    starts = np.r_[0, np.nonzero(np.diff(sp))[0] + 1, len(sp)]
    rows, cols = [], []
    for k in range(len(starts) - 1):
        nodes = sn[starts[k] : starts[k + 1]]
        if len(nodes) < 2:
            continue
        a, b = np.meshgrid(nodes, nodes, indexing="ij")
        u = a < b
        rows.append(a[u])
        cols.append(b[u])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    inter = coo_matrix(
        (np.ones(len(rows), np.int32), (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()
    inter.sum_duplicates()
    size = np.bincount(sn, minlength=n_nodes).astype(float)
    ii, jj = inter.nonzero()
    vals = np.asarray(inter[ii, jj]).ravel().astype(float)
    iou = vals / (size[ii] + size[jj] - vals)
    edge = (iou > seg["multiview_affinity_threshold"]) & (vals >= mv["min_edge_points"])
    edge &= node_view_arr[ii] != node_view_arr[jj]
    adj = coo_matrix((np.ones(edge.sum()), (ii[edge], jj[edge])), shape=(n_nodes, n_nodes))
    n_comp, comp = connected_components(adj, directed=False)

    # 5. Per-point vote over the clusters of the masks that see it.
    comp_of_obs = comp[obs_node]
    comp_size = np.bincount(comp, minlength=n_comp)
    multi = comp_size[comp_of_obs] >= mv["min_cluster_masks"]  # singleton masks don't vote
    key = obs_pt[multi].astype(np.int64) * n_comp + comp_of_obs[multi]
    uk, cnt = np.unique(key, return_counts=True)
    pt, cl = uk // n_comp, uk % n_comp
    o = np.lexsort((-cnt, pt))
    pt, cl, cnt = pt[o], cl[o], cnt[o]
    first = np.r_[True, pt[1:] != pt[:-1]]
    n_seen = np.bincount(obs_pt, minlength=n_pts)
    win_pt, win_cl, win_cnt = pt[first], cl[first], cnt[first]
    strong = (win_cnt >= mv["min_votes"]) & (win_cnt >= mv["min_vote_frac"] * n_seen[win_pt])
    point_comp = np.full(n_pts, -1, np.int64)
    point_comp[win_pt[strong]] = win_cl[strong]

    # 6. Area filter in the wall plane (footprint at the ortho GSD).
    spm = scale_params(seg, tf["metric"])
    gsd = spm["gsd"]
    cell = np.floor(p_wall[:, :2] / gsd).astype(np.int64)
    cell_key = (cell[:, 0] - cell[:, 0].min()) * (cell[:, 1].max() - cell[:, 1].min() + 1) + (
        cell[:, 1] - cell[:, 1].min()
    )
    has = point_comp >= 0
    pairs = np.unique(np.c_[point_comp[has], cell_key[has]], axis=0)
    area = np.bincount(pairs[:, 0], minlength=n_comp) * gsd**2
    ok_comp = (area >= spm["min_area"]) & (area <= spm["max_area"])
    lut = np.zeros(n_comp, np.int32)
    lut[ok_comp] = np.arange(1, ok_comp.sum() + 1)
    labels = np.zeros(n_pts, np.int32)
    labels[has] = lut[point_comp[has]]
    np.save(out / "points_labels.npy", labels)

    # Debug overlays.
    n_dbg = 0
    for vi, pv in enumerate(per_view):
        if pv["_lab"] is None:
            continue
        img = np.asarray(Image.open(mvs / "images" / views[vi]["name"]).convert("RGB"))
        ov = _overlay(img, pv["_lab"], pv["_rc"], labels[pv["_idx"]], seed)
        Image.fromarray(ov).save(out / "debug" / f"{Path(views[vi]['name']).stem}.jpg", quality=88)
        n_dbg += 1
    for pv in per_view:
        for k in ("_lab", "_rc", "_idx"):
            pv.pop(k)

    big = comp_size[ok_comp[np.arange(n_comp)]] if ok_comp.any() else np.array([0])
    return {
        "n_views": len(views),
        "n_masks": n_nodes,
        "n_edges": int(edge.sum()),
        "n_components": int(n_comp),
        "n_components_multi_view": int((comp_size >= mv["min_cluster_masks"]).sum()),
        "n_stones": int(ok_comp.sum()),
        "dropped_by_area": int(((~ok_comp) & (area > 0)).sum()),
        "max_masks_in_one_stone": int(big.max()),
        "labelled_point_fraction": float((labels > 0).mean()),
        "points_seen_by_any_mask": float((n_seen > 0).mean()),
        "sam_seconds": round(t_sam, 1),
        "n_debug_overlays": n_dbg,
        "per_view": per_view,
    }


def _debug_views(n: int, k: int) -> set[int]:
    return set(np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist())
