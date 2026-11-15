# Ground-truth stones for `highwall`

Needed for M7 (`evaluate`). One file: `stones_gt.json` in this folder, labelme format.

## Image to annotate

`outputs/highwall/06_segment/ortho/ortho.png` (3122 × 2455 px, wall frame, viewed from
the front). A copy of the orthophoto as annotated is kept here as `ortho_ref.jpg`;
`evaluate` registers each new orthophoto to it, so the annotation survives a
full rerun (D-019).

## Tool

[labelme](https://github.com/wkentaro/labelme), in its own environment so it does not
touch `stonetwin`:

Installed as a plain venv (no conda needed) at `D:\codingTools\venvs\labelme` (labelme 7.11).
From the repo root:

```
D:\codingTools\venvs\labelme\Scripts\labelme.exe outputs/highwall/06_segment/ortho/ortho.png --output data/annotations/highwall --labels region,stone --validate-label exact
```

labelme 7.x: `--output` is a folder and the file is named after the image (`ortho.json`);
image data is left out by default (the old `--nodata` flag no longer exists). When done,
rename `ortho.json` to `stones_gt.json`. Autosave is on, so reopening with the same
command resumes where you left off (rename only after you finish).

## What to draw

1. **One polygon labelled `region`** around a contiguous area of wall. Everything inside
   it is evaluated, so every stone inside must be drawn. Suggested area: the rubble
   masonry left of the blocked opening and below the large plaque, which is well
   reconstructed. Avoid the plaques, the iron door, the dark upper band and the edges.
2. **One polygon labelled `stone` per stone** inside the region, following the stone's
   visible face edge (where it meets mortar). At least **30** stones.
3. A stone cut by the region boundary: draw the whole stone anyway.
4. Leave mortar, patches, holes and non-stone objects unlabelled.

Save (Ctrl+S), then rename the resulting `ortho.json` to `data/annotations/highwall/stones_gt.json`. This file is committed;
it holds polygons only, no image data.
