"""Render a coloured point cloud to PNG with an orthographic z-buffer, for gate reports.

Not a pipeline stage. Pure numpy, so it does not depend on a working OpenGL context.

    python scripts/render_cloud.py cloud.ply out.png [--eye-dir dx dy dz] [--up ux uy uz]
        [--center x y z] [--extent E] [--width 1600] [--point-px 2] [--shade]

Coordinates: ``--center``, ``--extent`` (visible width), ``--eye-dir`` (direction from
the centre towards the viewer) and ``--up`` are in the cloud's own frame and units.
For ``dense_wall.ply`` (wall frame) the defaults look at the wall face along -z.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


def render(
    cloud: Path,
    out: Path,
    eye_dir: tuple[float, float, float] = (0, 0, 1),
    up: tuple[float, float, float] = (0, 1, 0),
    center: tuple[float, float, float] | None = None,
    extent: float | None = None,
    width: int = 1600,
    aspect: float = 1.6,
    point_px: int = 2,
    shade: bool = False,
) -> None:
    """Render ``cloud`` orthographically to ``out``.

    ``shade`` replaces colour by Lambertian shading of the normals, which shows
    geometry (joints, relief) independently of texture.
    """
    pcd = o3d.io.read_point_cloud(str(cloud))
    pts = np.asarray(pcd.points)
    d = np.asarray(eye_dir, float)
    d /= np.linalg.norm(d)
    u = np.asarray(up, float)
    u = u - d * (u @ d)
    u /= np.linalg.norm(u)
    r = np.cross(u, d)
    c = np.asarray(center, float) if center is not None else np.median(pts, axis=0)
    q = pts - c
    x, y, z = q @ r, q @ u, q @ d
    if extent is None:
        lo, hi = np.percentile(x, [0.5, 99.5])
        extent = float(hi - lo)
    height = int(round(width / aspect))
    s = width / extent
    px = np.round(x * s + width / 2).astype(int)
    py = np.round(-y * s + height / 2).astype(int)

    if shade and pcd.has_normals():
        n = np.asarray(pcd.normals)
        n = n * np.sign(n @ d)[:, None]
        # grazing light from upper left makes small relief (joints) visible
        light = 0.35 * d + 0.8 * u - 0.45 * r
        light /= np.linalg.norm(light)
        g = np.clip(n @ light, 0, 1)
        lo, hi = np.percentile(g, [2, 98])
        g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1) * 0.9 + 0.05
        col = np.repeat(g[:, None], 3, axis=1)
    elif pcd.has_colors():
        col = np.asarray(pcd.colors)
    else:
        col = np.full((len(pts), 3), 0.3)

    img = np.ones((height, width, 3))
    zbuf = np.full((height, width), -np.inf)
    order = np.argsort(z)  # far to near; nearer points overwrite
    half = point_px // 2
    for dy in range(-half, point_px - half):
        for dx in range(-half, point_px - half):
            xx, yy = px[order] + dx, py[order] + dy
            ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            xx, yy, zz, cc = xx[ok], yy[ok], z[order][ok], col[order][ok]
            # assignment with repeated indices keeps the last (= nearest) write
            closer = zz > zbuf[yy, xx]
            img[yy[closer], xx[closer]] = cc[closer]
            zbuf[yy[closer], xx[closer]] = zz[closer]
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cloud", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--eye-dir", type=float, nargs=3, default=(0, 0, 1))
    ap.add_argument("--up", type=float, nargs=3, default=(0, 1, 0))
    ap.add_argument("--center", type=float, nargs=3)
    ap.add_argument("--extent", type=float)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--aspect", type=float, default=1.6)
    ap.add_argument("--point-px", type=int, default=2)
    ap.add_argument("--shade", action="store_true")
    a = ap.parse_args()
    render(
        a.cloud,
        a.out,
        tuple(a.eye_dir),
        tuple(a.up),
        a.center,
        a.extent,
        a.width,
        a.aspect,
        a.point_px,
        a.shade,
    )


if __name__ == "__main__":
    main()
