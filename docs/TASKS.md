# Build Plan

Work top to bottom. Tick a box only when its acceptance check passes. At a **GATE**,
stop, write the gate report, and wait for the author.

Scene order: finish every milestone on `highwall` first. `entrance` only needs M1–M2
until Gate 2 is passed.

---

## Day 1 — geometry

### M0 Setup
- [x] Detect GPU, CUDA, conda, COLMAP, installed splat trainers. Write `docs/ENVIRONMENT.md`.
- [x] Choose the splat backend and the SAM-family model. Log both in `docs/DECISIONS.md`.
- [x] Create the `stonetwin` conda environment, `pyproject.toml`, `environment.yml`.
- [x] Scaffold the package and CLI with every stage as a stub that reads the config.
- [x] `git init`, first commit. Confirm `git status` shows nothing under `data/raw/` or `docs/internal/`.

**Accept:** `python -m stonetwin --help` lists all stages; `pytest` and `ruff check` run clean.

### M1 Ingest and SfM
- [x] `ingest` for both scenes; manifest and blur report written.
- [x] `sfm` for both scenes.

**Accept:** thresholds in `configs/default.yaml` under `sfm.accept`. If `highwall`
fails, try sequential matching and a lower blur rejection before anything else, and
log what changed.

### M2 Dense cloud and splat
- [x] `splat` for both scenes, with held-out metrics.
- [x] `dense` for `highwall`.
- [x] `frame` for `highwall` (runs without scale refs, flagged non-metric).

**Accept:** held-out PSNR / SSIM / LPIPS written; `dense_wall.ply` opens and shows
individual stones and joints when viewed up close.

### GATE 1 — reconstruction review
Write `outputs/gate1.md` with: registered fraction and reprojection error per scene,
held-out metrics, four screenshots of `dense_wall.ply` (overview, two close-ups, the
cavity), one splat render per scene, and a plain statement of whether joints between
stones are geometrically visible in the cloud.

---

## Day 2 — analysis

### M3 Registration experiment
- [x] Split `highwall` images, reconstruct both halves independently.
- [x] Coarse and fine alignment; write `T_b_to_a.json` and metrics.
- [x] Unit test for the similarity estimator.

**Accept:** ICP converges; RMSE and cloud-to-cloud statistics are in `stage.json`. If
a half fails to reconstruct with 21 images, switch the split to interleaved thirds
(two thirds against one third, overlapping coverage) and log it.

### M4 Segmentation
- [x] `ortho` method end to end. Produce the orthophoto first and hand it to the author for annotation.
- [x] `multiview` method end to end with debug overlays.

**Accept:** both methods write `points_labels.npy` under the same contract; overlays
exist; stone count is plausible against a manual count of one region.

### GATE 2 — segmentation review
Write `outputs/gate2.md` with: orthophoto, label maps from both methods side by side,
five multi-view debug overlays, stone counts, and the three worst failure regions with
a one-line diagnosis each. Recommend which method should be presented as the main
result.

### M5 Attributes
- [ ] `attributes` on the method chosen at Gate 2 (keep the other runnable).
- [ ] Unit test on a synthetic stone.
- [ ] Sanity check: the open cavity region shows strongly negative `recession`.

**Accept:** `stones.json` validates against the field table in the spec; no NaNs.

---

## Day 3 — delivery

### M6 Viewer
- [ ] `export-viewer`; payload under budget.
- [ ] Viewer with colour modes, picking and stone card.
- [ ] P1: splat layer.
- [ ] P1: GitHub Pages workflow (do not push; the author publishes).

**Accept:** clicking any stone opens the correct record; runs from a clean
`npm install && npm run build`.

### M7 Evaluation and report
- [ ] `evaluate` for both segmentation methods against the author's annotation.
- [ ] `report`: `metrics.json`, `docs/RESULTS.md`, figures, README results block.
- [ ] `python -m stonetwin all --scene highwall --force` runs clean from scratch.

**Accept:** every number in `README.md` and `docs/RESULTS.md` is traceable to
`metrics.json`.

### GATE 3 — release review
Write `outputs/gate3.md` with: the final metrics table, the privacy checklist (no GPS,
no serial numbers, no identifiable people in committed or deployed files, nothing from
`docs/internal/`), repository size, and a list of known limitations to carry into the
README.

---

## Backlog (only after Gate 3)

- [ ] F17 distance map figure
- [ ] F18 stone segmentation on `entrance`
- [ ] F19 second-epoch capture and comparison

## Waiting on the author

| Item | Needed by | File |
|---|---|---|
| Two measured reference distances | M5 (attributes in mm) | `data/meta/highwall/scale_refs.json` |
| ≥ 30 annotated stones | M7 | `data/annotations/highwall/stones_gt.json` |
| Site names | Gate 3 | `docs/DATA.md` |
