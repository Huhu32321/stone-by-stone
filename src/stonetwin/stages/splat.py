"""Stage ``splat``: 3D Gaussian Splatting with a held-out test split.

Backend ``lichtfeld`` (D-009): COLMAP ``image_undistorter`` writes a PINHOLE dataset to
``03_splat/data/``; LichtFeld Studio trains headless on it with ``--test-every
<holdout_every>``, so every N-th image is never trained on. At the final iteration it
writes side-by-side comparison images (ground truth | 4 px gap | render) for the
held-out views. This stage splits them into ``renders/test/<image>_gt.png`` and
``renders/test/<image>.png`` and computes PSNR, SSIM and LPIPS itself, so the metric
definitions do not depend on the trainer.

Frames: ``splat.ply`` is in the SfM ``world`` frame (non-metric), standard 3DGS PLY.
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from stonetwin.stage import AcceptanceError, StageContext, StageError
from stonetwin.stages.dense import _sfm_model
from stonetwin.tools import colmap_version, find_tool, run_logged

_GAP_PX = 4  # separator between GT and render in LichtFeld comparison images


def split_comparison(img: np.ndarray, gap: int = _GAP_PX) -> tuple[np.ndarray, np.ndarray]:
    """Split a LichtFeld eval image ``[H, 2W+gap, 3]`` into (ground truth, render)."""
    w = (img.shape[1] - gap) // 2
    if 2 * w + gap != img.shape[1]:
        raise StageError(f"unexpected comparison image width {img.shape[1]}")
    return img[:, :w], img[:, w + gap :]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two float images in [0, 1]."""
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


def _identify(gt: np.ndarray, data_images: Path, names: list[str]) -> str:
    """Return the dataset image whose downscaled copy best matches ``gt`` (uint8)."""
    h, w = gt.shape[:2]
    best, best_err = "", np.inf
    g = gt.astype(np.float32)
    for name in names:
        ref = Image.open(data_images / name).convert("RGB").resize((w, h), Image.BILINEAR)
        err = float(np.mean(np.abs(np.asarray(ref, np.float32) - g)))
        if err < best_err:
            best, best_err = name, err
    if best_err > 8.0:  # grey levels; a wrong match would be far larger
        raise StageError(f"could not identify held-out image (best error {best_err:.1f})")
    return best


def _metrics(test_dir: Path, names: list[str], lpips_net: str) -> dict:
    import lpips
    import torch
    from skimage.metrics import structural_similarity

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loss = lpips.LPIPS(net=lpips_net, verbose=False).to(dev)
    per_image = []
    for name in names:
        stem = Path(name).stem
        gt = np.asarray(Image.open(test_dir / f"{stem}_gt.png"), np.float32) / 255
        rd = np.asarray(Image.open(test_dir / f"{stem}.png"), np.float32) / 255
        ssim = structural_similarity(gt, rd, channel_axis=2, data_range=1.0)
        with torch.no_grad():
            t = [torch.from_numpy(x).permute(2, 0, 1)[None].to(dev) * 2 - 1 for x in (gt, rd)]
            lp = float(loss(t[0], t[1]).item())
        per_image.append({"image": name, "psnr": psnr(gt, rd), "ssim": float(ssim), "lpips": lp})
    return {
        "psnr": float(np.mean([r["psnr"] for r in per_image])),
        "ssim": float(np.mean([r["ssim"] for r in per_image])),
        "lpips": float(np.mean([r["lpips"] for r in per_image])),
        "per_image": per_image,
    }


def run(ctx: StageContext) -> None:
    """Train the splat, export ``splat.ply`` and held-out renders, compute metrics.

    Raises:
        StageError: tool missing or training failed.
        AcceptanceError: no held-out evaluation was produced.
    """
    scfg = dict(ctx.cfg["splat"])
    backend = "lichtfeld" if scfg["backend"] == "auto" else scfg["backend"]
    if backend != "lichtfeld":
        raise StageError(f"splat.backend '{backend}' is not implemented (D-009)")
    ctx.params = {**scfg, "backend_resolved": backend}
    colmap = find_tool("colmap", ctx.cfg)
    lfs = find_tool("lichtfeld", ctx.cfg)
    ctx.tools = {"colmap": colmap_version(colmap), "lichtfeld": lfs.name}
    model = _sfm_model(ctx)
    ctx.inputs = {"sfm_model": model.relative_to(ctx.scene_dir).as_posix()}

    for sub in ("data", "lichtfeld", "renders"):
        if (ctx.out_dir / sub).exists():  # stale run; inside out_dir only
            shutil.rmtree(ctx.out_dir / sub)
    log = ctx.out_dir / "splat.log"
    log.unlink(missing_ok=True)
    data = ctx.out_dir / "data"
    run_logged(
        [
            colmap,
            "image_undistorter",
            "--image_path",
            ctx.root / ctx.cfg["raw_dir"],
            "--input_path",
            model,
            "--output_path",
            data,
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(scfg["max_image_size"]),
        ],
        log,
    )
    for extra in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
        p = data / extra
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)

    train_dir = ctx.out_dir / "lichtfeld"
    run_logged(
        [
            lfs,
            "--headless",
            "--train",
            "-d",
            data,
            "-o",
            train_dir,
            "-i",
            str(scfg["iterations"]),
            "--test-every",
            str(scfg["holdout_every"]),
            "--resize_factor",
            "1",
            "--max-cap",
            str(scfg["max_gaussians"]),
            "--eval",
            "--save-eval-images",
            "--log-level",
            "warn",
        ],
        log,
    )

    plys = sorted(train_dir.glob("splat_*.ply"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    evals = sorted(train_dir.glob("eval_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[-1]))
    if not plys:
        raise StageError(f"LichtFeld wrote no splat PLY in {train_dir}; see {log}")
    if not evals:
        raise AcceptanceError(f"LichtFeld wrote no held-out evaluation in {train_dir}")
    shutil.copy2(plys[-1], ctx.out_dir / "splat.ply")
    final_eval = evals[-1]

    test_dir = ctx.out_dir / "renders" / "test"
    test_dir.mkdir(parents=True)
    all_names = sorted(p.name for p in (data / "images").iterdir())
    names: list[str] = []
    for f in sorted(final_eval.glob("*.png"), key=lambda p: int(p.stem)):
        gt, rd = split_comparison(np.asarray(Image.open(f).convert("RGB")))
        name = _identify(gt, data / "images", all_names)
        names.append(name)
        Image.fromarray(gt).save(test_dir / f"{Path(name).stem}_gt.png")
        Image.fromarray(rd).save(test_dir / f"{Path(name).stem}.png")

    m = _metrics(test_dir, names, scfg["lpips_net"])
    trainer_csv = train_dir / "metrics.csv"
    trainer_rows = list(csv.DictReader(trainer_csv.open())) if trainer_csv.is_file() else []
    ctx.metrics = {
        "n_train": len(all_names) - len(names),
        "n_test": len(names),
        "test_images": names,
        "eval_iteration": int(final_eval.name.rsplit("_", 1)[-1]),
        "render_resolution": list(gt.shape[1::-1]),
        **m,
        "n_gaussians": int(trainer_rows[-1]["num_gaussians"]) if trainer_rows else None,
        "trainer_reported": trainer_rows[-1] if trainer_rows else None,
    }
