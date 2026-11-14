# Stone by Stone

**A per-stone digital twin of a masonry wall from handheld photographs.**

Photogrammetry and 3D Gaussian Splatting show what a wall looks like. This project
goes one step further and asks which stone is which: it reconstructs a real rubble
masonry wall, segments it into individual stones in 3D, measures each one, and puts
the result in a web viewer where every stone can be clicked.

> **Status: in development.** The pipeline is specified and the data is captured.
> Results below are filled in by the pipeline when it runs; nothing here is typed by
> hand.

<!-- Add after first results: docs/figures/teaser.png and the live viewer link -->

## What it does

| Stage | Method | Output |
|---|---|---|
| Reconstruct | Structure-from-Motion, dense point cloud, 3D Gaussian Splatting | Camera poses, point cloud, splat |
| Frame | RANSAC wall-plane fit, scale from measured reference distances | Metric wall coordinate frame |
| Register | Two independent reconstructions from disjoint image subsets, aligned by a 7-DoF similarity transform (feature-based coarse step, then ICP) | Transform and residual |
| Parse | Foundation-model masks lifted to 3D by multi-view voting, compared with a 2.5D orthophoto baseline | A stone ID for every point |
| Describe | Per-stone geometry and colour statistics | `stones.json` |
| Verify | Hand-annotated ground truth, Hungarian matching | IoU, precision, recall |
| Show | Static three.js viewer | Click a stone, read its record |

## Results

<!-- RESULTS:START -->
_Not yet generated. Run `python -m stonetwin report --scene highwall`._
<!-- RESULTS:END -->

Full tables and figures: [`docs/RESULTS.md`](docs/RESULTS.md).

## Scenes

| Scene | Subject | Images | Used for |
|---|---|---|---|
| `highwall` | Rubble masonry wall with repairs and an open cavity | 42 | Full pipeline and evaluation |
| `entrance` | Free-standing pointed stone arch | 155 | Reconstruction showcase |

Both were captured in Dublin with a Canon EOS R8 at 24 mm. See the
[dataset card](docs/DATA.md) for capture conditions and known problems.

## Per-stone record

```json
{
  "id": 17,
  "centroid_wall": [0.0, 0.0, 0.0],
  "face_area": 0.0,
  "width": 0.0,
  "height": 0.0,
  "roughness_rms": 0.0,
  "recession": 0.0,
  "colour_lab": [0.0, 0.0, 0.0],
  "colour_delta_e": 0.0
}
```

The values above are a schema illustration. `roughness_rms`, `recession` and
`colour_delta_e` are condition proxies computed from geometry and colour. They are not
a classification of deterioration and have not been validated against expert survey.

## Reproduce

```bash
conda env create -f environment.yml
conda activate stonetwin

# place photos in data/raw/<scene>/
python -m stonetwin all --scene highwall

cd viewer && npm install && npm run dev
```

Each stage can be run alone, for example `python -m stonetwin segment --scene highwall`.
Outputs land in `outputs/<scene>/`, one folder per stage, each with a `stage.json`
recording parameters, tool versions and metrics.

## Limitations

- Image-only capture: no laser scanning and no surveyed control. Scale comes from
  tape-measured reference distances.
- The registration experiment splits one capture session in two. It tests the
  estimator, not change over time.
- One wall, one annotator, a few dozen ground-truth stones.
- The segmentation model is used as released, with no fine-tuning on masonry.

## Repository layout

```
configs/        pipeline defaults and per-scene overrides
data/           measurements and annotations (raw photos are not in git)
docs/           requirements, technical spec, dataset card, decisions, results
src/stonetwin/  pipeline stages
tests/          unit tests for geometry and metrics
viewer/         web viewer
```

## Documentation

- [Requirements](docs/PRD.md)
- [Technical specification](docs/TECH_SPEC.md)
- [Dataset card](docs/DATA.md)
- [Design decisions](docs/DECISIONS.md)

## Related work

The orthophoto baseline follows the 2.5D idea used for rubble masonry by Valero,
Bosché and Forster (*Automation in Construction*, 2018). Segmenting stones in images
and transferring them to point clouds for stone-level analysis is explored by Tobies
et al. (*ISPRS Annals*, 2026).

## Licence

Code: [MIT](LICENSE). Photographs © Yuhui Cao.

## Author

Yuhui Cao, M.Sc. Computer Science, Trinity College Dublin.
