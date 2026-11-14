# Design Decisions

A running log. Newest at the bottom. Each entry: what was decided, why, and what was
rejected. Claude Code appends here whenever it makes or changes a choice.

---

### D-001 `highwall` is the analysis scene, `entrance` is the showcase
**Date:** 2026-11-14 · **By:** product

`highwall` is rubble masonry with distinct joints, a patched area and an open cavity,
so it exercises segmentation and the condition proxies. It is also planar, which
permits the orthophoto baseline. `entrance` has more images and a more striking shape
but regular, tightly jointed blocks, thin railings and many occluders, so it is used
for reconstruction only.

*Rejected:* running the full pipeline on both. Not affordable in three days.

### D-002 Two segmentation methods, baseline first
**Date:** 2026-11-14 · **By:** product

The orthophoto method is simple and nearly certain to work on a planar wall. The
multi-view method is the more general contribution and the riskier one. Building the
baseline first guarantees a shippable result and gives the multi-view method something
to be compared against.

*Rejected:* multi-view only. One failure would leave nothing to show.

### D-003 Registration is tested on independent half-reconstructions
**Date:** 2026-11-14 · **By:** product

Aligning the dense cloud to the splat would be trivial because both share the SfM
frame. Splitting the images and reconstructing each half separately produces two
clouds with different origin, orientation and scale, which makes the similarity
estimate a real problem with a meaningful residual.

*Rejected:* ICP between products of the same SfM run. *Deferred:* a true second epoch
(F19).

### D-004 Condition attributes are proxies, and named as such
**Date:** 2026-11-14 · **By:** product

With no expert labels and no material measurements, any claim about deterioration type
or severity would be unsupported. The project reports roughness, recession and colour
deviation, and says what they are.

### D-005 Raw photographs stay out of git
**Date:** 2026-11-14 · **By:** product

About 440 MB, with GPS and camera metadata in EXIF. The repository carries code,
documentation, annotations and the downsampled viewer payload only.

### D-006 Pipeline runs without scale and flags it
**Date:** 2026-11-14 · **By:** product

No scale bar was placed during capture. Rather than block on a site revisit, every
output carries `metric: true | false`. Attributes are reported in millimetres only
once reference distances have been entered.

### D-007 Python 3.11 and PyTorch built for CUDA 12.8
**Date:** 2026-11-14 · **By:** claude-code

The GPU is an RTX 5070 Ti Laptop (compute capability 12.0, Blackwell). Only PyTorch
wheels built against CUDA 12.8 or later contain kernels for it, so the environment uses
the `cu128` wheel index. Python 3.11 is the project minimum and is inside the range
that open3d ships wheels for.

*Rejected:* Python 3.12/3.13 (no benefit, narrower wheel support for open3d); older
CUDA builds of PyTorch (do not run on this GPU).

### D-008 COLMAP 4.2.1 official Windows CUDA binary, plus pycolmap for I/O
**Date:** 2026-11-14 · **By:** claude-code

No COLMAP was installed. The official release binary provides GPU SIFT and
`patch_match_stereo`, which enables `dense.backend: colmap_mvs`. It is installed outside
the repository; its path is given by the `STONETWIN_COLMAP` environment variable or
`configs/local.yaml`. `pycolmap` is used only to read models. If the binary's MVS does
not run on this GPU, `dense` falls back to `splat_depth_fusion` as the spec allows.

*Rejected:* pycolmap only (CPU-only on Windows, no MVS); RealityScan (proprietary,
not reproducible by a reviewer).

*Update (same day, M0 smoke test):* SfM works on the GPU, but `patch_match_stereo`
aborts on this machine: its PTX was built with a newer CUDA toolchain than driver
591.66 supports (details in `ENVIRONMENT.md`). Tried 4.2.1 and 4.1.1; stopped after
two attempts. Until the driver is updated, `dense.backend: auto` must resolve to
`splat_depth_fusion`. The auto check will run a one-view MVS probe and fall back.

### D-009 Splat backend: LichtFeld Studio (installed)
**Date:** 2026-11-14 · **By:** claude-code

LichtFeld Studio v0.5.0 is already installed. It is a CUDA 3DGS trainer with a headless
CLI that reads a COLMAP model, supports `--test-every` for a held-out split and writes
a standard 3DGS PLY. This matches the spec's first preference (a trainer already on the
machine). PSNR, SSIM and LPIPS are computed by `stonetwin` itself from the held-out
renders, so the metric definitions do not depend on the trainer.

*Rejected:* `gsplat` (needs a CUDA toolkit for JIT compilation, which is not installed;
kept as fallback).

### D-010 Segmentation model: SAM 2.1 Hiera-Large, automatic mask generation
**Date:** 2026-11-14 · **By:** claude-code

SAM 2.1 provides a maintained automatic mask generator with openly downloadable
checkpoints. It is installed with its optional CUDA post-processing extension disabled
(`SAM2_BUILD_CUDA=0`), which needs no compiler. Checkpoint goes in `checkpoints/`
(git-ignored).

*Rejected:* SAM 3 (concept/text-prompted, gated weights, less suited to "every
stone" mask generation); original SAM ViT-H (older, no advantage here).

### D-011 Blur flagging by rank; capture order from EXIF time
**Date:** 2026-11-14 · **By:** claude-code

`blur_reject_percentile` is applied as a rank: the `round(n * p / 100)` images with the
lowest Laplacian variance are flagged (2 of 42 for `highwall`, 8 of 155 for `entrance`
at 5 %). A percentile threshold on the score would flag a varying number of images when
scores tie. Capture order (used by the `odd_even` split in M3) is EXIF
`DateTimeOriginal` with the file name as tie-break; the camera clock is wrong in
absolute terms but consistent within a session.

*Rejected:* a fixed absolute blur threshold (scenes differ by an order of magnitude in
median score: about 320 vs 610).

### D-012 Dense backend: COLMAP MVS at 1600 px, fusion with 3 agreeing views
**Date:** 2026-11-14 · **By:** claude-code

After the NVIDIA driver was updated to 621.24 (CUDA 13.5), `patch_match_stereo` runs on
this GPU (it aborted under 591.66, see D-008). `dense.backend: auto` therefore resolves
to `colmap_mvs`. A 1200 px probe on `highwall` took 12 min for 40 views and showed the
wall, plaques and cavity clearly. The stage uses 1600 px (`dense.max_image_size`) as a
compromise between joint detail and run time (full 2400 px would be roughly four times
slower). `stereo_fusion` keeps points seen consistently in 3 views instead of COLMAP's
default 5: on the probe this gave 54 % more points (1.33 M vs 0.86 M) and filled gaps at
the wall edges, with no visible increase in noise on the wall face.

`splat_depth_fusion` is not implemented, since the preferred backend now works. The
stage fails with a clear message if it is selected. The MVS workspace is kept in
`02_dense/mvs/` so the `multiview` method can reuse its depth maps for the visibility
test.

*Rejected:* full-resolution MVS (time); fusion default of 5 views (sparser at the
edges of a 40-image capture).

### D-013 Splat metrics computed from LichtFeld's held-out comparison images
**Date:** 2026-11-14 · **By:** claude-code

LichtFeld trains on a COLMAP-undistorted PINHOLE copy of the registered images
(`splat.max_image_size: 1600`) with `--test-every 8`, `--resize_factor 1`, and writes
ground-truth|render pairs for the held-out views at the last iteration. `stonetwin`
splits those pairs and computes PSNR, SSIM (scikit-image) and LPIPS (VGG backbone, as
in the original 3DGS evaluation) itself. On a 600-iteration smoke test our PSNR matched
LichtFeld's own value to 4 decimals, SSIM differed by 0.003 (window implementation).

Held-out images are identified by matching each ground-truth half against the dataset
images, not by assuming an index pattern: on the smoke test LichtFeld held out images
1, 10, 19, 28, 37 of the sorted list, not 0, 8, 16, … . `--max-cap 3000000` bounds
VRAM.

*Rejected:* writing a separate renderer (needs `gsplat` and a CUDA toolkit, D-009);
trusting the trainer's metrics alone (definitions not documented, no LPIPS).

### D-014 Wall frame ROI in scene units until scale arrives
**Date:** 2026-11-14 · **By:** claude-code

The `frame` ROI keeps points from `roi_band_back` behind to `roi_band_front` in front
of the fitted plane, inside the convex hull of where the central 80 % of each image
meets the plane. Without scale references these bands are in scene units, chosen from
the `highwall` reconstruction (wall about 11 scene units wide, wall relief within
±0.05, cavity to about −0.3): 0.4 behind, 0.15 in front. A first try with 1.0 / 0.5
kept a wide strip of pavement; 0.15 still keeps a narrow one, plus a short piece of the
return wall at the left end. They must be revisited
in metres once `scale_refs.json` exists. RANSAC runs on a 300 k-point subsample and the
plane is then refined by least squares over all inliers.

Ruff's pep8-naming rules N802/N803 are ignored in addition to N806, because the
project's frame-naming convention (`T_wall_from_world`, `apply_T`) requires capitals.

---

<!-- Template
### D-0NN Title
**Date:** YYYY-MM-DD · **By:** claude-code | author | product

What and why.

*Rejected:* alternatives.
-->
