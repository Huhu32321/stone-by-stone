# Environment

Detected on 2026-11-14 during M0. Updated whenever a tool is added or upgraded.
Machine-local install paths are not recorded here; they are set through environment
variables or `configs/local.yaml` (git-ignored).

## Hardware

| | |
|---|---|
| OS | Windows 11 Home, build 26200, x64 |
| CPU | AMD Ryzen 9 8945HX |
| RAM | 31 GB |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GB, compute capability 12.0 (Blackwell) |
| Driver | 591.66 (supports CUDA runtime up to 13.1) |
| Disk | Project drive has ample space; system drive has about 21 GB free |

Compute capability 12.0 needs CUDA 12.8 or later. Any PyTorch or CUDA binary built for
older architectures only will not run on this GPU.

## Tools found before setup

| Tool | Version | Status |
|---|---|---|
| conda (Miniconda) | 26.3.2 | present; base env is Python 3.13 |
| git | 2.55.0 | present |
| Node.js / npm | 24.15.0 / 11.12.1 | present (for the viewer) |
| Visual Studio 2022 Community | with C++ build tools | present |
| CUDA toolkit (nvcc) | — | **not installed** (driver only) |
| COLMAP | — | **not installed** |
| LichtFeld Studio | v0.5.0-85-g4ad2d1a4 | present; CUDA 3DGS trainer with headless CLI |
| RealityScan | 2.1.1 | present; proprietary, not used by the pipeline |
| Python on PATH | — | only the Microsoft Store alias; use conda |

## Project environment

Conda environment `stonetwin`, Python 3.11.18. Exact versions are in
[`environment.yml`](../environment.yml). Key packages:

| Package | Version | Check |
|---|---|---|
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 | CUDA available, device (12, 0), matmul on GPU OK |
| SAM 2.1 (`SAM-2`) | git `2b90b9f5ceec`, checkpoint `sam2.1_hiera_large.pt` | automatic mask generation on a 1200×800 photo: 89 masks, 2.5 s, 3.2 GB peak GPU memory |
| open3d | 0.20.0 | |
| pycolmap | 4.2.1 | model I/O only |
| opencv-python | 5.0.0.93 | |
| shapely | 2.2.0 | |
| lpips | 0.1.4 | |

## External tools

Paths are set in `configs/local.yaml` (git-ignored; template in
`configs/local.example.yaml`).

| Tool | Version | Status on this machine |
|---|---|---|
| COLMAP | 4.2.1, official `windows-cuda` build | GPU SIFT extraction, exhaustive matching, mapping and undistortion work. **`patch_match_stereo` aborts** (exit 0xC0000409). |
| LichtFeld Studio | v0.5.0-85-g4ad2d1a4 | headless training to be exercised in M2 |

### COLMAP MVS on this GPU

Smoke test on six `highwall` images (copies in a temporary folder): every stage up to
undistortion succeeds; `patch_match_stereo` aborts on every view and writes no depth
maps. COLMAP 4.2.1 aborts silently. COLMAP 4.1.1 (tested once for comparison, then
removed) prints the cause:

```
CUDA error at src/colmap/mvs/patch_match_cuda.cu:1683
- the provided PTX was compiled with an unsupported toolchain.
```

The PatchMatch kernel is shipped as PTX built with a newer CUDA toolkit than driver
591.66 can JIT-compile. A newer NVIDIA driver would most likely fix it; until then the
`dense` stage uses `splat_depth_fusion` (see D-008).
