"""SAM 2.1 automatic mask generation (D-010), shared by both segmentation methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stonetwin.stage import StageError

_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def load_generator(cfg: dict[str, Any]) -> Any:
    """Build a ``SAM2AutomaticMaskGenerator`` from ``cfg['segment']['sam']``.

    The checkpoint path is ``cfg['tools']['sam2_checkpoint']`` relative to the repo root.
    """
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    ckpt = Path(cfg["_root"]) / cfg["tools"]["sam2_checkpoint"]
    if not ckpt.is_file():
        raise StageError(f"SAM 2 checkpoint not found: {ckpt}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(_MODEL_CFG, str(ckpt), device=dev, apply_postprocessing=False)
    sc = dict(cfg["segment"]["sam"])
    return SAM2AutomaticMaskGenerator(model, **sc)


def generate(gen: Any, image_rgb: np.ndarray) -> list[dict[str, Any]]:
    """Run automatic mask generation on an RGB uint8 image (H, W, 3).

    Returns SAM's records: ``segmentation`` (bool H x W), ``area`` (px),
    ``predicted_iou``, ``stability_score``, ``bbox`` (x, y, w, h).
    """
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return gen.generate(image_rgb)
