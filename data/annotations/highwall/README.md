# Ground-truth stones for `highwall`

Needed for M7 (`evaluate`). One file: `stones_gt.json` in this folder, labelme format.

## Image to annotate

`outputs/highwall/06_segment/ortho/ortho.png` (3122 × 2455 px, wall frame, viewed from
the front). Its pixel geometry is in `ortho_meta.json` next to it. The annotation stays
valid as long as `dense` and `frame` are not rerun (D-016).

## Tool

[labelme](https://github.com/wkentaro/labelme), in its own environment so it does not
touch `stonetwin`:

```
conda create -n labelme python=3.11 -y
conda activate labelme
pip install labelme
labelme outputs/highwall/06_segment/ortho/ortho.png --output data/annotations/highwall/stones_gt.json --nodata
```

`--nodata` keeps the image out of the JSON (it would otherwise embed the picture).

## What to draw

1. **One polygon labelled `region`** around a contiguous area of wall. Everything inside
   it is evaluated, so every stone inside must be drawn. Suggested area: the rubble
   masonry left of the blocked opening and below the large plaque, which is well
   reconstructed. Avoid the plaques, the iron door, the dark upper band and the edges.
2. **One polygon labelled `stone` per stone** inside the region, following the stone's
   visible face edge (where it meets mortar). At least **30** stones.
3. A stone cut by the region boundary: draw the whole stone anyway.
4. Leave mortar, patches, holes and non-stone objects unlabelled.

Save (Ctrl+S) to `data/annotations/highwall/stones_gt.json`. This file is committed;
it holds polygons only, no image data.
