"""Privacy checks for Gate 3. Not a pipeline stage.

Checks, over files tracked by git (and the built viewer, if present):

0. Report how many raw photos carry a camera serial number, and GPS in EXIF or XMP.
1. No tracked path under ``data/raw/`` or ``docs/internal/``.
2. No tracked image carries EXIF (or any GPS / serial tag).
3. The camera serial numbers and GPS coordinates found in the raw photos' EXIF do not
   appear as text in any tracked file (several number formats are searched).
4. No tracked text file quotes a line from ``docs/internal/``.

The secret values are never printed; only whether they were found and where.

    python scripts/privacy_check.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import ExifTags, Image

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def tracked() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [ROOT / p for p in out.stdout.decode().split("\0") if p]


def _dms_to_deg(v) -> float:
    d, m, s = (float(x) for x in v)
    return d + m / 60 + s / 3600


def raw_secrets() -> tuple[set[str], dict[str, int]]:
    """Serial numbers and GPS coordinate strings from the raw photos (never printed),
    plus counts of photos with a serial number and with GPS in EXIF or XMP."""
    secrets: set[str] = set()
    counts = {"raw_photos": 0, "with_serial": 0, "with_gps_exif": 0, "with_gps_xmp": 0}
    tags = {v: k for k, v in ExifTags.TAGS.items()}
    for p in sorted((ROOT / "data" / "raw").rglob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        counts["raw_photos"] += 1
        if b"GPSLatitude" in p.read_bytes()[:262144]:
            counts["with_gps_xmp"] += 1
        exif = Image.open(p).getexif()
        ifd = exif.get_ifd(0x8769)
        if ifd.get(tags["BodySerialNumber"]):
            counts["with_serial"] += 1
        for key in ("BodySerialNumber", "LensSerialNumber", "CameraSerialNumber"):
            v = ifd.get(tags[key]) or exif.get(tags.get(key, -1))
            if v and str(v).strip("\0 ").strip():
                secrets.add(str(v).strip("\0 ").strip())
        gps = exif.get_ifd(0x8825)
        if 2 in gps and 4 in gps:
            counts["with_gps_exif"] += 1
            for v in (_dms_to_deg(gps[2]), _dms_to_deg(gps[4])):
                for nd in (3, 4, 5):
                    secrets.add(f"{v:.{nd}f}")
    return {s for s in secrets if len(s) >= 5}, counts


def main() -> int:
    files = tracked()
    problems: list[str] = []

    bad_paths = [
        p
        for p in files
        if p.relative_to(ROOT).as_posix().startswith(("data/raw/", "docs/internal/"))
    ]
    problems += [f"tracked private path: {p.relative_to(ROOT)}" for p in bad_paths]

    images = [p for p in files if p.suffix.lower() in IMG_EXT]
    dist = ROOT / "viewer" / "dist"
    if dist.is_dir():
        images += [p for p in dist.rglob("*") if p.suffix.lower() in IMG_EXT]
    for p in images:
        exif = Image.open(p).getexif()
        if len(exif):
            problems.append(f"image has EXIF ({len(exif)} tags): {p.relative_to(ROOT)}")

    secrets, counts = raw_secrets()
    internal_lines = set()
    for p in (ROOT / "docs" / "internal").glob("*"):
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if len(line) > 40 and not line.startswith(("#", "|", "-")):
                internal_lines.add(line)
    n_text = 0
    for p in files:
        if p.suffix.lower() in IMG_EXT or p.suffix.lower() in {".bin", ".ply", ".npy"}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        n_text += 1
        if any(s in text for s in secrets):
            problems.append(f"raw EXIF serial/GPS value appears in: {p.relative_to(ROOT)}")
        if any(line in text for line in internal_lines):
            problems.append(f"quotes docs/internal: {p.relative_to(ROOT)}")

    print(
        f"tracked files: {len(files)}; images checked: {len(images)}; "
        f"text files scanned: {n_text}; secret strings from raw EXIF: {len(secrets)}"
    )
    print("raw photo metadata:", counts)
    for line in problems:
        print("PROBLEM:", line)
    print("PASS" if not problems else f"FAIL ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
