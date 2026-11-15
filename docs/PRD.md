# Product Requirements — Stone by Stone

| | |
|---|---|
| Owner | Yuhui Cao |
| Status | Approved for build |
| Version | 1.0 (2026-11-14) |
| Build budget | 3 working days |

## 1. One-line summary

Turn handheld photographs of a real stone wall into a per-stone digital twin: every
stone gets an ID, a 3D footprint and measured attributes, and can be clicked in a web
viewer.

## 2. Problem

Conservation of stone-built heritage is decided stone by stone, but most 3D captures
stop at an undifferentiated surface. A splat or a point cloud shows what a wall looks
like; it does not say which stone is which, how large it is, or which ones are
receding. This project closes that gap at small scale, with consumer equipment and a
pipeline that can be re-run and checked.

## 3. Users

| User | What they need from this repo |
|---|---|
| Reviewer with a geodesy / photogrammetry background | Sound method, honest numbers, stated limits. Reads `README.md` and `docs/RESULTS.md`, opens the viewer. Has five minutes. |
| The author | A working demo and a clear account of every design choice. |
| Claude Code | Unambiguous tasks, interfaces and acceptance criteria. Reads `CLAUDE.md` first. |

## 4. Goals

1. **G1 Reconstruct.** Recover camera geometry and a dense 3D representation of each
   scene (point cloud and 3D Gaussian Splatting).
2. **G2 Register.** Solve a real 7-DoF registration between two independent
   reconstructions of the same wall and report its residual.
3. **G3 Parse.** Segment the wall into individual stones in 3D.
4. **G4 Describe.** Compute per-stone geometric and photometric attributes and export
   them as a structured record.
5. **G5 Show.** Publish an interactive viewer where each stone is selectable.
6. **G6 Verify.** Report every quality claim as a number produced by a script, against
   manually annotated ground truth where one is needed.

## 5. Non-goals

- Validated classification of stone deterioration type or severity. The attributes are
  proxies and are labelled as such everywhere.
- Survey-grade accuracy. There is no laser scanner and no control network.
- IFC / full HBIM authoring. A JSON record per stone is the deliverable.
- Training a new segmentation model. A pretrained foundation model is used as is.
- Real-time or in-headset XR delivery.

## 6. Scenes

| Scene | Role | Images | Priority |
|---|---|---|---|
| `highwall` | Analysis scene. Rubble masonry with clear joints, a patched area and an open cavity. | 42 | **P0** for every goal |
| `entrance` | Showcase scene. Free-standing pointed stone arch. | 155 | P0 for G1, P2 for G3–G4 |

Details and known data risks are in [`DATA.md`](DATA.md).

## 7. Requirements

Priority: **P0** must ship, **P1** should ship, **P2** only if time remains.

### Functional

| ID | Requirement | Pri |
|---|---|---|
| F1 | Ingest a folder of photos, extract EXIF, score blur, write a manifest. Raw files are never modified. | P0 |
| F2 | Run SfM and produce camera poses and a sparse cloud for each scene. | P0 |
| F3 | Produce a dense point cloud for `highwall`. | P0 |
| F4 | Train a 3D Gaussian Splatting model per scene with a held-out test split. | P0 |
| F5 | Define a wall coordinate frame (along-wall, up, outward) by plane fitting. | P0 |
| F6 | Run in relative units and flag `metric: false` throughout (no scale reference in the capture; D-020). | P0 |
| F7 | Register two independent reconstructions built from disjoint image subsets (similarity transform: coarse then ICP) and report RMSE. | P0 |
| F8 | Segment stones with the orthophoto baseline (2.5D). | P0 |
| F9 | Segment stones by lifting per-image masks to 3D with multi-view voting. | P0 |
| F10 | Compute per-stone attributes and write `stones.json` and `stones.csv`. | P0 |
| F11 | Web viewer: orbit the point cloud, colour by RGB / stone ID / any attribute, click a stone to open its record. | P0 |
| F12 | Evaluate segmentation against at least 30 hand-annotated stones. | P0 |
| F13 | Generate `docs/RESULTS.md` and README figures from `metrics.json`. No hand-typed numbers. | P0 |
| F14 | One command reproduces a scene end to end. | P0 |
| F15 | Splat layer in the viewer, toggled against the point cloud. | P1 |
| F16 | Deploy the viewer to GitHub Pages. | P1 |
| F17 | Cloud-to-cloud distance map between the two registered reconstructions. | P1 |
| F18 | Stone segmentation on `entrance`. | P2 |
| F19 | Second-epoch capture of `highwall` and true two-epoch comparison. | P2 |

### Quality

| ID | Requirement |
|---|---|
| Q1 | Every stage reads a config, writes to its own output folder and can be re-run alone. |
| Q2 | Fixed seeds; tool names and versions recorded in `docs/ENVIRONMENT.md`. |
| Q3 | No GPS coordinates, camera serial numbers or identifiable bystanders in anything committed or deployed. |
| Q4 | Code passes `ruff` and the unit tests for geometry utilities. |
| Q5 | The README can be understood without reading any other file. |

## 8. Success metrics

Targets are what we aim for, not what we will claim. Whatever the scripts measure is
what gets reported, including results below target.

| Metric | Scene | Target |
|---|---|---|
| Images registered by SfM | both | ≥ 90 % |
| Mean reprojection error | both | < 1.0 px |
| Held-out PSNR / SSIM / LPIPS | both | reported |
| Registration RMSE after ICP | highwall | reported, in relative units |
| Stone segmentation mean IoU (matched) | highwall | ≥ 0.60 |
| Stone detection F1 at IoU 0.5 | highwall | ≥ 0.70 |
| Multi-view method vs orthophoto baseline | highwall | both reported side by side |

## 9. Deliverables

1. Public GitHub repository with this structure and a results-bearing README.
2. Live viewer link.
3. `docs/RESULTS.md` with tables and figures generated from metrics.
4. A 60-second screen recording of the viewer (made by the author, not in scope for
   Claude Code).

## 10. Milestones and gates

Build order and acceptance criteria are in [`TASKS.md`](TASKS.md). There are three
review gates where work stops until the author signs off:

- **Gate 1** after reconstruction: is the geometry good enough to analyse?
- **Gate 2** after segmentation: is the multi-view method working, or do we ship the
  baseline as the main result?
- **Gate 3** before publishing: are the numbers and the privacy checks in order?

## 11. Risks

| Risk | Likelihood | Response |
|---|---|---|
| 42 images are too few for a clean dense cloud of `highwall` | Medium | Check at Gate 1. Fall back to depth fusion from the splat; if still poor, recapture. |
| `highwall` was shot at ISO 1000–5000; noise weakens matching and colour attributes | Medium | Report it. Use colour attributes with caution and say so. |
| No scale reference was placed in the scene | High | Accepted: results are relative and say so (F6, D-020). |
| Multi-view lifting is unstable | Medium | The orthophoto baseline (F8) is built first and is always shippable. |
| Railings, lamp post and passers-by occlude `entrance` | High | Reconstruction only for this scene; mask or accept floaters. |
| Time overrun | Medium | Cut in this order: F18, F19, F17, F15. Never cut F12 or F13. |

## 12. Open items for the author

- [x] Confirm the site names and write them into `docs/DATA.md`.
- [x] Annotate at least 30 stones on the orthophoto once M4 produces it.
- [ ] Decide the public repository name (suggested: `stone-by-stone`).
