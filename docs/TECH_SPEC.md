# Technical Specification

This document fixes interfaces, outputs and acceptance checks. Where it says
"choose in M0", make the choice from what is installed, then record it in
[`DECISIONS.md`](DECISIONS.md). Library versions and APIs move quickly: check current
documentation before relying on remembered signatures.

## 1. Pipeline

```
photos ─► ingest ─► sfm ─┬─► dense ─┐
                         └─► splat ─┤
                                    ▼
                                  frame ─► register
                                    │
                                    ▼
                                 segment ─► attributes ─► evaluate ─► report
                                                │
                                                └─► export-viewer
```

## 2. Package and CLI

Python package `stonetwin` under `src/`. One entry point, one sub-command per stage:

```
python -m stonetwin <stage> --scene <name> [--config configs/default.yaml]
python -m stonetwin all --scene highwall
```

Stages: `ingest`, `sfm`, `dense`, `splat`, `frame`, `register`, `segment`,
`attributes`, `evaluate`, `export-viewer`, `report`.

Rules for every stage:

- Reads the merged config (`configs/default.yaml` + `configs/scenes/<scene>.yaml`).
- Writes only inside `outputs/<scene>/<NN_stage>/`.
- Writes `stage.json` there with: inputs, parameters, tool versions, wall-clock time,
  and its metrics.
- Exits non-zero with a clear message if an acceptance check fails.
- Is idempotent; `--force` recomputes.

## 3. Output layout

```
outputs/<scene>/
  00_ingest/     manifest.csv, blur_report.png, stage.json
  01_sfm/        colmap/ (database, sparse model), cameras.json, stage.json
  02_dense/      dense.ply, stage.json
  03_splat/      splat.ply, renders/test/, stage.json
  04_frame/      transform.json, wall_plane.json, dense_wall.ply, stage.json
  05_register/   subset_a.ply, subset_b.ply, T_b_to_a.json, c2c.png, stage.json
  06_segment/    ortho/ (ortho.png, ortho_meta.json, labels.png, points_labels.npy)
                 multiview/ (masks/, points_labels.npy, debug/)
                 stage.json
  07_attributes/ stones.json, stones.csv, stage.json
  08_evaluate/   matches.csv, overlays/, stage.json
  metrics.json   merged metrics from every stage.json
```

## 4. Stage specifications

### 4.0 ingest

- Read every image in `raw_dir`. Record file name, size, EXIF capture time, focal
  length, ISO, exposure, f-number.
- Blur score: variance of the Laplacian on a grayscale copy. Flag the lowest
  `blur_reject_percentile`. Flagged images are excluded from later stages through the
  manifest (`use = false`); the files stay where they are.
- Fail if focal length or resolution is not constant while `sfm.single_camera` is true.
- Never write into `data/raw/`.

### 4.1 sfm

- COLMAP: feature extraction, exhaustive matching, incremental mapping. One shared
  `OPENCV` camera per scene.
- Export poses and intrinsics to `cameras.json` (world-from-camera, plus K and
  distortion), so later stages do not parse COLMAP binaries.
- Metrics: registered fraction, mean reprojection error, number of 3D points, mean
  track length.
- Acceptance: thresholds in `sfm.accept`.

### 4.2 dense

- `colmap_mvs` when a CUDA-enabled COLMAP is available. Otherwise
  `splat_depth_fusion`: render depth from the trained splat at training poses and fuse
  with a TSDF (Open3D), which makes this stage depend on 4.3.
- Output: `dense.ply` with positions, normals and RGB.
- Metrics: point count, median nearest-neighbour spacing.

### 4.3 splat

- Backend chosen in M0. Preference order: a trainer already installed on the machine,
  then `gsplat`-based training. Input is the COLMAP model from 4.1.
- Hold out every `splat.holdout_every`-th image. Report PSNR, SSIM and LPIPS on the
  held-out set only.
- Output: `splat.ply` in the standard 3DGS PLY layout.

### 4.4 frame

- Fit the dominant wall plane to the dense cloud with RANSAC, then refine by least
  squares on inliers.
- Wall frame: `x` along the wall (horizontal), `y` up, `z` outward from the wall
  toward the cameras. Up comes from the mean camera up-vector projected into the
  plane. Origin at the inlier centroid.
- Scale: the capture has no scale reference, so scale is 1 and `metric: false` is
  written into `transform.json` and propagated to every later output (D-020).
- Crop to a wall region of interest (points within a distance band of the plane, inside
  the convex hull of where the cameras were looking). Save as `dense_wall.ply`.
- `freeform` scenes skip plane fitting and only apply gravity alignment.

### 4.5 register

This stage is the parameter-estimation experiment. Both clouds must come from
independent reconstructions, so they do not share a gauge.

1. Split usable images into two disjoint subsets (`odd_even` by capture order).
2. Run SfM and dense reconstruction on each subset separately. Each result has its own
   arbitrary origin, orientation and scale.
3. Coarse alignment: FPFH features with RANSAC, estimating a similarity transform.
4. Fine alignment: point-to-plane ICP, with scale re-estimated (Umeyama on
   correspondences) between ICP rounds.
5. Report: estimated scale, rotation angle, translation, inlier RMSE, fitness, and the
   distribution of cloud-to-cloud distances (median, 95th percentile).
6. P1: colour the cloud by signed distance and save `c2c.png`.

State plainly in the report that this is one capture session split in two. It tests
the estimator, not change over time.

### 4.6 segment

Two methods, same output contract: `points_labels.npy`, an integer per point of
`dense_wall.ply`, where 0 means mortar or unassigned and 1..N are stone IDs.

**Method `ortho` (baseline, build first).**

1. Rasterise `dense_wall.ply` orthographically along `-z` at `ortho_gsd` into an RGB
   orthophoto and a depth (relief) map. Keep the point-to-pixel index.
2. Run automatic mask generation with a pretrained SAM-family model on the orthophoto.
   Model and checkpoint chosen in M0.
3. Filter masks by area bounds, drop masks that contain other masks almost entirely
   (keep the finer level), resolve overlaps by assigning each pixel to the smallest
   containing mask.
4. Write `labels.png` and map back to points.

**Method `multiview` (main).**

1. Run the same mask generator on every registered image.
2. For each point in `dense_wall.ply`, project into each image. Keep the observation
   only if the point passes a depth test against that view's depth map (from MVS or
   rendered from the splat).
3. Each mask now owns a set of 3D point indices. Build a graph whose nodes are masks.
   Connect two masks from different images when the IoU of their point sets exceeds
   `multiview_affinity_threshold`.
4. Cluster the graph (connected components first; switch to a community method only
   if components over-merge). Each cluster is one stone.
5. Label each point by majority vote over the clusters of the masks that see it.
   Points with weak support stay 0.
6. Apply the same area filter in the wall plane.

Write debug overlays for at least five views. Segmentation failures are far easier to
diagnose from pictures than from numbers.

### 4.7 attributes

One record per stone. Lengths and areas in scene units (relative), with
`metric: false` at the top of the file.

| Field | Definition |
|---|---|
| `id` | Stone ID from segmentation |
| `n_points` | Supporting points |
| `centroid_wall` | Centroid in the wall frame `[x, y, z]` |
| `face_area` | Area of the concave hull of the points projected on the wall plane |
| `width`, `height` | Extents of the minimum-area bounding rectangle in the wall plane |
| `aspect_ratio` | `width / height` |
| `roughness_rms` | RMS residual to the stone's own best-fit plane |
| `recession` | Median offset of the stone's face from the local wall surface, defined as the median `z` of neighbouring stones. Negative means set back. |
| `colour_lab` | Mean CIELAB colour |
| `colour_std` | Standard deviation of L\* |
| `colour_delta_e` | ΔE between the stone's mean colour and the scene median |
| `outline_wall` | Hull polygon in wall coordinates, for the viewer |

`roughness_rms`, `recession` and `colour_delta_e` are **condition proxies**. They are
not deterioration classes and must never be labelled as such in code, the viewer or
the README.

### 4.8 evaluate

- Ground truth: polygons drawn by the author on the orthophoto, saved as
  `data/annotations/<scene>/stones_gt.json` (labelme format). At least 30 stones in one
  contiguous region, so that missed stones inside the region count as errors.
- Restrict evaluation to the annotated region.
- Match predictions to ground truth with the Hungarian algorithm on IoU.
- Report for each method: mean IoU of matched pairs, precision, recall and F1 at
  `iou_match_threshold`, over-segmentation count (one ground-truth stone covered by
  several predictions) and under-segmentation count.
- Save an overlay image per method.

### 4.9 export-viewer

- Downsample to `viewer.max_points`. Write a compact binary of positions, RGB and
  stone ID, plus `stones.json`, into `viewer/public/data/<scene>/`.
- Strip all EXIF from any image copied into the viewer or the docs.
- Fail if the payload exceeds `viewer.max_payload_mb`.

### 4.10 report

- Merge every `stage.json` into `outputs/<scene>/metrics.json`.
- Regenerate `docs/RESULTS.md` and the figures in `docs/figures/` from it.
- Rewrite the block between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->` in
  `README.md`.

## 5. Viewer

- Static site in `viewer/`: Vite with three.js. No backend.
- Point cloud rendered as points; a picking pass (ID colour buffer) resolves the
  clicked stone.
- Controls: scene selector, colour mode (RGB / stone ID / attribute), attribute
  dropdown with a legend and value range, orbit and pan.
- Stone card: ID, dimensions, area, the three condition proxies with units, and a
  short note that the proxies are not validated classifications.
- P1: load the splat as a second layer with a toggle. Choose the splat rendering
  library in M6 after checking which ones are currently maintained.
- P1: GitHub Pages deployment through an Actions workflow.

## 6. Environment

- Windows, x64. Detect the GPU, CUDA version, conda, COLMAP and any installed splat
  trainer in M0 and write `docs/ENVIRONMENT.md`.
- One conda environment named `stonetwin`. `pyproject.toml` declares dependencies;
  `environment.yml` pins the solved versions.
- Expected libraries: numpy, scipy, open3d, opencv-python, scikit-image, shapely,
  torch, a SAM-family package, pycolmap or the COLMAP CLI, typer, pyyaml, pytest,
  ruff.

## 7. Tests

Unit tests for the parts that are easy to get subtly wrong:

- Similarity transform estimation recovers a known scale, rotation and translation.
- Projection and depth test on a synthetic camera.
- Plane fit and wall frame axes on a synthetic tilted plane.
- Polygon IoU and Hungarian matching on hand-built cases.
- Attribute values on a synthetic stone of known size.
