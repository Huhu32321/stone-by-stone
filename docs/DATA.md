# Dataset Card

Facts in this file were read from the image files on 2026-11-14. Anything not
measurable from the files is marked **to confirm**.

## Capture summary

| | `highwall` | `entrance` |
|---|---|---|
| Subject | Rubble masonry wall with a plaque, a patched area and an open cavity exposing loose stone | Free-standing pointed stone arch with iron railings, on cobbled paving |
| Site | **to confirm** | **to confirm** |
| Images | 42 | 155 |
| Camera | Canon EOS R8 | Canon EOS R8 |
| Lens | RF 24-105mm F4-7.1 IS STM at 24 mm | same, at 24 mm |
| Resolution | 2400 × 1600 | 2400 × 1600 |
| Aperture | f/4 – f/4.5 | f/4 |
| Shutter | 1/60 s | 1/60 – 1/320 s |
| ISO | 1000 – 5000 | 125 – 800 |
| Capture span | about 2.5 minutes | about 16.5 minutes, one frame every 5 s |
| Capture date | 14 November 2026 | 14 November 2026 |
| Scale reference in frame | none placed | none placed |

The camera clock was not set to local time. Use EXIF timestamps for ordering only.

## Known issues

1. **No scale bar.** Scale must come from distances measured on site afterwards. See
   `data/meta/highwall/scale_refs.example.json`.
2. **`highwall` is short and noisy.** 42 frames at high ISO. Expect weaker matching
   and noisier colour than `entrance`.
3. **`entrance` has motion blur.** Frames were taken on a 5-second interval while
   moving; at least one sampled frame is visibly blurred. The ingest blur filter
   matters here.
4. **`entrance` has occluders.** Railings, a lamp post, foliage, sky and at least one
   passer-by. Thin railings and sky will produce floaters.
5. **Downsized JPEGs.** Files are 2400 × 1600, not the sensor's native resolution. If
   originals exist, re-export at full resolution before a second pass.

## Capture protocol for a second epoch

To make the next capture comparable and metric:

- Overcast daylight, base ISO where possible, shutter 1/125 s or faster.
- Fixed focal length, fixed focus mode, no zooming between frames.
- Three passes parallel to the wall (level, tilted up, tilted down) with at least 70 %
  overlap, plus a few oblique frames at each end.
- Two scale bars or a tape fixed in the scene, visible in at least five frames each.
- 80–120 frames for a 5 m section.
- Note the weather and time of day.

## Privacy and publishing

- Raw images carry the camera body serial number and other camera metadata in EXIF.
  No GPS coordinates were found in EXIF or XMP of any of the 197 files (checked
  2026-11-15 with `scripts/privacy_check.py`); treat them as private regardless.
  They are excluded from git by `.gitignore`.
- Any image copied into `docs/` or `viewer/` must have EXIF stripped and must not show
  an identifiable person.
- If a sample of the raw data is published, publish an EXIF-stripped copy as a release
  asset, not in the repository history.

## Folder layout

```
data/
  raw/<scene>/            original photos, read-only, not in git
  meta/<scene>/           scale_refs.json and other hand-entered measurements
  annotations/<scene>/    ground-truth stone polygons
```
