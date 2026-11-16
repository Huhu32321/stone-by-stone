"""English diagrams and charts for a presentation. Not a pipeline stage.

Writes ``docs/presentation/*.svg`` (editable) and a results chart (``.svg`` + ``.png``).
Every number drawn comes from ``outputs/highwall/metrics.json``; nothing is typed.
PNGs of the diagrams are made by ``viewer/tests/svg_to_png.mjs`` (headless Chrome).

    python scripts/presentation_figures.py
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation"

W, H = 1600, 900
INK, QUIET, LINE = "#1f2328", "#57606a", "#8c959f"
ACCENT, ACCENT_BG = "#b5651d", "#fbefe3"
GROUP_BG, ENGINE, ENGINE_BG = "#f6f8fa", "#3b6ea8", "#eef4fb"
FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"


class Svg:
    """Tiny SVG builder: boxes with a title and body lines, arrows, labels."""

    def __init__(self, title: str, subtitle: str = "") -> None:
        self.parts: list[str] = []
        self.text(60, 72, title, 34, INK, bold=True)
        if subtitle:
            self.text(60, 110, subtitle, 20, QUIET)

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: int = 18,
        fill: str = INK,
        bold: bool = False,
        anchor: str = "start",
        italic: bool = False,
    ) -> None:
        w = ' font-weight="600"' if bold else ""
        it = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}"{w}{it} '
            f'text-anchor="{anchor}">{escape(s)}</text>'
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "white",
        stroke: str = LINE,
        sw: float = 1.5,
        dash: str = "",
        rx: int = 10,
    ) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        lines: tuple[str, ...] = (),
        kind: str = "plain",
        center: bool = False,
    ) -> None:
        fill, stroke, sw = {
            "plain": ("white", LINE, 1.5),
            "accent": (ACCENT_BG, ACCENT, 2.5),
            "engine": (ENGINE_BG, ENGINE, 1.5),
            "group": (GROUP_BG, "#d0d7de", 1.2),
        }[kind]
        self.rect(x, y, w, h, fill, stroke, sw)
        if center and not lines:
            self.text(x + w / 2, y + h / 2 + 8, title, 21, INK, True, "middle")
            return
        tx = x + w / 2 if center else x + 18
        anchor = "middle" if center else "start"
        self.text(tx, y + 34, title, 21, INK, True, anchor)
        for i, line in enumerate(lines):
            self.text(tx, y + 62 + i * 25, line, 17, QUIET, anchor=anchor)

    def arrow(
        self,
        pts: list[tuple[float, float]],
        label: str = "",
        lx: float = 0,
        ly: float = 0,
        dash: str = "",
        color: str = LINE,
        anchor: str = "middle",
    ) -> None:
        d = "M" + " L".join(f"{x} {y}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        mk = "arrA" if color == ACCENT else ("arrE" if color == ENGINE else "arr")
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"'
            f'{da} marker-end="url(#{mk})"/>'
        )
        if label:
            self.text(lx, ly, label, 16, QUIET, anchor=anchor, italic=True)

    def save(self, name: str) -> Path:
        defs = "".join(
            f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" '
            f'fill="{c}"/></marker>'
            for i, c in (("arr", LINE), ("arrA", ACCENT), ("arrE", ENGINE))
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
            f'height="{H}" font-family="{FONT}"><defs>{defs}</defs>'
            f'<rect width="{W}" height="{H}" fill="white"/>' + "".join(self.parts) + "</svg>"
        )
        p = OUT / f"{name}.svg"
        p.write_text(svg, encoding="utf-8")
        return p


def f(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


# --------------------------------------------------------------------------- diagrams
def architecture(m: dict) -> Path:
    s = Svg(
        "System architecture",
        "One Python package drives external engines; every stage leaves a stage.json",
    )
    # Inputs
    s.text(60, 168, "INPUTS", 16, QUIET, True)
    s.box(
        60,
        182,
        300,
        130,
        "Photographs",
        ("highwall: 42 · entrance: 155", "Canon EOS R8, 24 mm", "read-only, never in git"),
    )
    s.box(
        60,
        332,
        300,
        104,
        "Hand annotation",
        (
            f"{m['evaluate']['metrics']['n_gt_in_region']} stone polygons",
            "labelme, on the orthophoto",
        ),
    )
    s.box(
        60,
        456,
        300,
        104,
        "Configuration",
        ("default.yaml + one file per scene", "every parameter in YAML"),
    )
    # Pipeline container
    s.rect(420, 150, 760, 560, GROUP_BG, "#d0d7de", 1.2)
    s.text(444, 186, "stonetwin  ·  python -m stonetwin all --scene highwall", 19, INK, True)
    groups = [
        ("Reconstruct", ("ingest · sfm", "dense · splat")),
        ("Frame & register", ("frame · register",)),
        ("Parse & describe", ("segment (ortho, multiview)", "attributes")),
        ("Verify & deliver", ("evaluate · export-viewer", "report")),
    ]
    for i, (t, lines) in enumerate(groups):
        gx, gy = 444 + (i % 2) * 362, 206 + (i // 2) * 132
        s.box(gx, gy, 346, 116, t, lines, "accent" if t == "Parse & describe" else "plain")
    s.text(444, 488, "Each stage reads the merged config, writes only its own folder,", 16, QUIET)
    s.text(444, 510, "and records inputs, parameters, tool versions, time and metrics.", 16, QUIET)
    # Engines
    s.text(444, 552, "EXTERNAL ENGINES (called by the stages)", 16, QUIET, True)
    eng = [
        ("COLMAP 4.2.1", "SfM, MVS (GPU)"),
        ("LichtFeld Studio", "3DGS training"),
        ("SAM 2.1 Hiera-L", "mask generation"),
        ("Open3D · OpenCV", "geometry, ICP"),
    ]
    for i, (t, sub) in enumerate(eng):
        s.box(
            444 + (i % 2) * 362,
            566 + (i // 2) * 66,
            346,
            54,
            f"{t}  ·  {sub}",
            (),
            "engine",
            center=True,
        )
    # Outputs
    s.text(1240, 168, "OUTPUTS", 16, QUIET, True)
    s.box(
        1240,
        182,
        300,
        130,
        "outputs/<scene>/",
        ("one folder per stage", "stage.json in each", "metrics.json merged"),
    )
    s.box(
        1240,
        332,
        300,
        130,
        "Generated docs",
        ("docs/RESULTS.md, figures", "README results block", "no number typed by hand"),
    )
    s.box(
        1240,
        482,
        300,
        130,
        "Web viewer",
        (
            "Vite + three.js, static",
            f"{m['export-viewer']['metrics']['n_points']:,} points",
            f"{m['export-viewer']['metrics']['payload_mb']} MB payload",
        ),
        "accent",
    )
    s.box(1240, 632, 300, 78, "GitHub Pages", ("manual workflow",))
    # Arrows
    for y in (247, 384, 508):
        s.arrow([(360, y), (420, y)])
    s.arrow([(1180, 247), (1240, 247)])
    s.arrow([(1180, 397), (1240, 397)])
    s.arrow([(1180, 547), (1240, 547)])
    s.arrow([(1390, 612), (1390, 632)])
    s.text(
        60,
        760,
        "Privacy: raw photos, EXIF serial numbers and private notes never reach git "
        "or the viewer (automated check).",
        17,
        QUIET,
    )
    s.text(
        60,
        800,
        "Lengths are relative scene units: the capture contains no scale reference.",
        17,
        QUIET,
    )
    return s.save("01_architecture")


def pipeline(m: dict) -> Path:
    s = Svg(
        "Pipeline: 11 stages, one command",
        "Photos in, clickable stones and an auto-generated report out",
    )
    bw, bh = 236, 118
    xs = [60, 336, 612, 888, 1164]
    r1, r2, r3 = 170, 400, 630

    def b(x, y, t, lines, kind="plain"):
        s.box(x, y, bw, bh, t, lines, kind)

    sfm = m["sfm"]["metrics"]
    b(xs[0], r1, "0 · ingest", ("EXIF, blur score", "manifest.csv"))
    b(
        xs[1],
        r1,
        "1 · sfm",
        ("COLMAP camera poses", f"reproj. {f(sfm['mean_reproj_error_px'], 2)} px"),
    )
    b(
        xs[2],
        r1,
        "2 · dense",
        ("COLMAP MVS", f"{m['dense']['metrics']['n_points'] / 1e6:.1f} M points"),
    )
    b(xs[3], r1, "4 · frame", ("wall plane (RANSAC)", "x along, y up, z out"))
    b(xs[4], r1, "6 · segment", ("ortho + multiview", "stone ID per point"), "accent")
    b(xs[1], r2, "3 · splat", ("3D Gaussian Splatting", "held-out PSNR/SSIM/LPIPS"))
    b(xs[2], r2, "5 · register", ("two independent halves", "similarity + ICP"))
    b(xs[4], r2, "7 · attributes", ("size, roughness,", "recession, colour"))
    b(xs[3], r2, "8 · evaluate", ("vs 231 hand-drawn", "stones, Hungarian IoU"))
    b(xs[3], r3, "9 · export-viewer", ("1.5 M points + stone", "records for the web"))
    b(xs[0], r3, "10 · report", ("metrics.json, RESULTS,", "figures, README"))
    b(xs[4], r3, "Ground truth", ("author's polygons", "on the orthophoto"))
    mid = bh / 2
    s.arrow([(xs[0] + bw, r1 + mid), (xs[1], r1 + mid)])
    s.arrow([(xs[1] + bw, r1 + mid), (xs[2], r1 + mid)])
    s.arrow([(xs[2] + bw, r1 + mid), (xs[3], r1 + mid)])
    s.arrow([(xs[3] + bw, r1 + mid), (xs[4], r1 + mid)])
    s.arrow([(xs[1] + bw / 2, r1 + bh), (xs[1] + bw / 2, r2)])
    s.arrow(
        [(xs[1] + bw, r1 + bh - 20), (xs[2] + 40, r2)],
        "split photos",
        xs[2] - 8,
        r2 - 60,
        anchor="end",
    )
    s.arrow([(xs[4] + bw / 2, r1 + bh), (xs[4] + bw / 2, r2)])
    s.arrow([(xs[4], r2 + mid), (xs[3] + bw, r2 + mid)])
    s.arrow(
        [
            (xs[4], r3 + mid),
            (xs[3] + bw + 30, r3 + mid),
            (xs[3] + bw + 30, r2 + bh - 20),
            (xs[3] + bw, r2 + bh - 20),
        ],
        dash="6 5",
        color=ACCENT,
    )
    s.arrow([(xs[3] + bw / 2, r2 + bh), (xs[3] + bw / 2, r3)])
    gap_y = r2 + bh + 56
    s.arrow(
        [(xs[3] + 40, r2 + bh), (xs[3] + 40, gap_y), (xs[0] + bw / 2, gap_y), (xs[0] + bw / 2, r3)]
    )
    s.text(xs[0] + bw / 2 + 12, gap_y - 10, "metrics from every stage", 16, QUIET, italic=True)
    s.text(
        60,
        820,
        "Every stage writes outputs/<scene>/<NN_stage>/ and a stage.json; "
        "--force recomputes, otherwise finished stages are skipped.",
        17,
        QUIET,
    )
    return s.save("02_pipeline")


def multiview(m: dict) -> Path:
    mv = m["segment"]["metrics"]["multiview"]
    s = Svg(
        "Multi-view lifting: 2D masks voted into 3D stones",
        "Each 3D point is labelled by the photos that actually see it",
    )
    bw, bh, y = 232, 160, 200
    steps = [
        ("1 · Masks", (f"SAM 2.1 on {mv['n_views']} photos", f"{mv['n_masks']:,} masks")),
        ("2 · Visibility", ("project wall points;", "keep if MVS depth", "agrees within 2 %")),
        ("3 · Mask graph", ("node = a mask's points", "edge = IoU > 0.5,", "different photos")),
        ("4 · Components", ("connected masks", "= one stone", f"{mv['n_edges']:,} edges")),
        ("5 · Point vote", ("majority of the views", "that see the point", "≥ 2 votes, ≥ 40 %")),
        ("6 · Area filter", ("footprint on the", "wall plane", f"→ {mv['n_stones']} stones")),
    ]
    for i, (t, lines) in enumerate(steps):
        x = 60 + i * (bw + 18)
        s.box(x, y, bw, bh, t, lines, "accent" if i in (1, 2) else "plain")
        if i:
            s.arrow([(x - 18, y + bh / 2), (x, y + bh / 2)])
    # Baseline comparison
    s.rect(60, 430, 1480, 310, GROUP_BG, "#d0d7de", 1.2)
    s.text(84, 476, "Baseline for comparison: ortho", 22, INK, True)
    s.text(
        84,
        512,
        "Rasterise the wall cloud into one orthophoto  →  SAM 2.1 once  →  filter by "
        "area, drop masks that contain a finer one  →  map pixels back to points.",
        18,
        QUIET,
    )
    o = m["segment"]["metrics"]["ortho"]
    e = m["evaluate"]["metrics"]
    s.text(84, 566, "Why multi-view wins here", 20, INK, True)
    s.text(
        84,
        600,
        f"• Points labelled: {f(mv['labelled_point_fraction'], 2)} (multiview) vs "
        f"{f(o['labelled_point_fraction'], 2)} (ortho)",
        18,
        INK,
    )
    s.text(
        84,
        632,
        f"• Recall against the annotation: {f(e['multiview']['recall'], 2)} vs "
        f"{f(e['ortho']['recall'], 2)}; F1 {f(e['multiview']['f1'], 3)} "
        f"vs {f(e['ortho']['f1'], 3)}",
        18,
        INK,
    )
    s.text(
        84,
        664,
        "• A joint that is faint in one photo is often clear in another; the "
        "orthophoto is a single, partly inpainted image.",
        18,
        INK,
    )
    s.text(
        84,
        708,
        "Known weakness: two-tone stones are split consistently across views, and "
        "SAM is class-agnostic (plaques, a door).",
        18,
        QUIET,
    )
    return s.save("03_multiview_lifting")


def registration(m: dict) -> Path:
    r = m["register"]["metrics"]
    ref = r["camera_reference"]
    s = Svg(
        "Registration experiment: two independent reconstructions",
        "A real similarity-estimation problem: the halves share no origin, orientation or scale",
    )
    s.box(60, 170, 220, 110, "40 photos", ("split by capture", "order: odd / even"))
    s.box(340, 150, 260, 110, "Half A · 20 photos", ("own SfM + MVS", "own gauge"))
    s.box(340, 290, 260, 110, "Half B · 20 photos", ("own SfM + MVS", "own gauge"))
    s.box(
        660,
        220,
        280,
        130,
        "Coarse",
        ("normalise each cloud", "FPFH features + RANSAC", "with scale (7-DoF)"),
    )
    s.box(
        1000,
        220,
        300,
        130,
        "Fine",
        (
            "point-to-plane ICP, scale,",
            "rotation, translation solved",
            f"jointly: {r['icp_iterations']} iterations",
        ),
        "accent",
    )
    s.box(
        1360,
        220,
        180,
        130,
        "T(A←B)",
        (f"scale {f(r['scale'], 4)}", f"rot. {f(r['rotation_angle_deg'], 2)}°"),
        center=True,
    )
    s.arrow([(280, 225), (340, 205)])
    s.arrow([(280, 225), (340, 345)])
    s.arrow([(600, 205), (660, 260)])
    s.arrow([(600, 345), (660, 310)])
    s.arrow([(940, 285), (1000, 285)])
    s.arrow([(1300, 285), (1360, 285)])
    # Reference branch
    s.box(
        340,
        470,
        260,
        110,
        "Full reconstruction",
        ("all 40 cameras", "(never seen by ICP)"),
        "engine",
    )
    s.box(
        660,
        470,
        280,
        110,
        "Camera-pose reference",
        ("Umeyama on each half's", "camera centres"),
        "engine",
    )
    s.arrow([(600, 525), (660, 525)], color=ENGINE)
    s.arrow([(940, 525), (1450, 525), (1450, 350)], color=ENGINE, dash="6 5")
    s.text(1460, 450, "compare", 16, ENGINE, italic=True)
    # Results
    s.rect(60, 630, 1480, 200, GROUP_BG, "#d0d7de", 1.2)
    s.text(84, 672, "Result vs. the camera-pose reference", 22, INK, True)
    s.text(
        84,
        708,
        f"Scale error {f(100 * ref['scale_rel_error'], 2)} %   ·   rotation error "
        f"{f(ref['rotation_error_deg'], 3)}°   ·   translation error "
        f"{f(100 * ref['translation_error_rel_extent'], 3)} % of the scene size",
        19,
        INK,
    )
    s.text(
        84,
        742,
        f"Cloud-to-cloud in the overlap: median {f(r['c2c_overlap']['median'], 4)}, "
        f"95th pct {f(r['c2c_overlap']['p95'], 4)} (point spacing "
        f"{f(r['a_median_nn_spacing'], 4)}) — relative units",
        19,
        INK,
    )
    s.text(
        84,
        784,
        "Finding: a smooth bowl a similarity cannot absorb. Holding the lens calibration "
        "fixed shrinks it below half: mostly per-half calibration.",
        19,
        ACCENT,
        True,
    )
    s.text(
        84,
        812,
        "Caveat: one session split in two tests the estimator, not change over time.",
        17,
        QUIET,
    )
    return s.save("04_registration")


def evaluation(m: dict) -> Path:
    e = m["evaluate"]["metrics"]
    ar = e["annotation_registration"]
    s = Svg(
        "Evaluation protocol",
        "The annotation is drawn once and re-registered to every run's orthophoto",
    )
    bw, bh, y = 270, 150, 200
    steps = [
        (
            "Ground truth",
            (f"{e['n_gt_in_region']} stones drawn by", "hand on the orthophoto", "(labelme)"),
        ),
        (
            "Re-register",
            (
                "SIFT + RANSAC similarity",
                f"{ar['n_inliers']:,} inliers,",
                f"{f(ar['rms_px_full_res'], 2)} px RMS",
            ),
        ),
        ("Region", ("union of stones, joints", "closed; undrawn plaques,", "door, arch stay out")),
        (
            "Hungarian matching",
            ("one-to-one on IoU", "true positive at", f"IoU ≥ {e['iou_threshold']}"),
        ),
        ("Scores", ("precision, recall, F1", "mean IoU of matches", "over- / under-segmentation")),
    ]
    for i, (t, lines) in enumerate(steps):
        x = 60 + i * (bw + 32)
        s.box(x, y, bw, bh, t, lines, "accent" if i == 1 else "plain")
        if i:
            s.arrow([(x - 32, y + bh / 2), (x, y + bh / 2)])
    s.rect(60, 420, 1480, 300, GROUP_BG, "#d0d7de", 1.2)
    s.text(84, 462, "Why the re-registration step exists", 22, INK, True)
    lines = [
        "Without an absolute scale, every SfM run has its own scale and origin, so the "
        "orthophoto grid moves between runs.",
        "Mapping polygons through wall coordinates was tried first and failed: mean IoU of "
        "matches dropped for both methods at once.",
        "A shared drop in both methods pointed at the ground truth, not the segmentation; "
        "image registration fixed it.",
        "Predictions count if at least half their area lies in the region; they are clipped "
        "to it before matching.",
    ]
    for i, t in enumerate(lines):
        s.text(84, 506 + i * 40, "• " + t, 18, INK)
    s.text(
        84,
        688,
        "Thresholds are fixed in the config; F1 at IoU 0.3 and 0.7 is reported for "
        "information only.",
        17,
        QUIET,
    )
    return s.save("05_evaluation")


def wall_frame(m: dict) -> Path:
    fr = m["frame"]["metrics"]
    s = Svg(
        "Wall coordinate frame",
        "From an arbitrary SfM gauge to a frame where z means 'proud of' or "
        "'set back from' the wall",
    )
    # Illustration: wall plane as a parallelogram, axes
    ox, oy = 470, 520
    s.parts.append(
        f'<polygon points="{ox - 330},{oy + 150} {ox + 230},{oy + 150} {ox + 330},{oy - 230} '
        f'{ox - 230},{oy - 230}" fill="{ACCENT_BG}" stroke="{ACCENT}" stroke-width="2"/>'
    )
    s.text(ox - 220, oy - 190, "fitted wall plane", 18, ACCENT, True)
    s.text(ox - 220, oy - 164, "RANSAC + least squares", 18, ACCENT)
    s.arrow([(ox, oy), (ox + 250, oy)], color=INK)
    s.arrow([(ox, oy), (ox + 66, oy - 250)], color=INK)
    s.arrow([(ox, oy), (ox - 150, oy + 230)], color=INK)
    s.text(ox + 125, oy - 16, "x  along the wall", 20, INK, True, "middle")
    s.text(ox + 80, oy - 258, "y  up", 20, INK, True)
    s.text(ox - 150, oy + 262, "z  out, toward the cameras", 20, INK, True, "middle")
    s.parts.append(f'<circle cx="{ox}" cy="{oy}" r="6" fill="{INK}"/>')
    s.text(ox + 14, oy + 28, "origin: inlier centroid", 16, QUIET)
    # Text column
    x0 = 960
    s.text(x0, 200, "How each axis is defined", 22, INK, True)
    items = [
        "z: plane normal, flipped to face the mean camera position.",
        "y: mean camera 'up' direction, projected into the plane.",
        "x: y × z, so the frame is right-handed.",
        "Scale 1: lengths stay in relative scene units.",
    ]
    for i, t in enumerate(items):
        s.text(x0, 240 + i * 38, "• " + t, 18, INK)
    s.text(x0, 420, "Region of interest", 22, INK, True)
    s.text(x0, 458, "• A band around the plane, wide enough behind it", 18, INK)
    s.text(x0, 486, "  for the open cavity.", 18, INK)
    s.text(x0, 520, "• Inside the hull where the central 80 % of each", 18, INK)
    s.text(x0, 548, "  photo meets the plane.", 18, INK)
    s.text(x0, 610, "This run", 22, INK, True)
    s.text(x0, 648, f"• {fr['n_points_wall']:,} points kept in the wall region", 18, INK)
    s.text(x0, 680, f"• plane inlier RMS {f(fr['inlier_rms'], 4)}", 18, INK)
    s.text(
        x0,
        712,
        f"• wall region {f(fr['wall_extent'][0], 1)} × {f(fr['wall_extent'][1], 1)} units",
        18,
        INK,
    )
    s.text(
        60,
        840,
        "Every later stage (segmentation, recession, the viewer) works in this frame.",
        17,
        QUIET,
    )
    return s.save("06_wall_frame")


def results_chart(m: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = m["evaluate"]["metrics"]
    thr = sorted(e["multiview"]["f1_by_iou_threshold"], key=float)
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(16, 9), dpi=100, gridspec_kw={"width_ratios": [3, 2]}
    )
    fig.patch.set_facecolor("white")
    xs = range(len(thr))
    wd = 0.36
    for k, (meth, col, lab) in enumerate(
        (("ortho", "#9aa4ae", "ortho (baseline)"), ("multiview", ACCENT, "multiview (main)"))
    ):
        vals = [e[meth]["f1_by_iou_threshold"][t] for t in thr]
        bars = a1.bar([x + (k - 0.5) * wd for x in xs], vals, wd, color=col, label=lab)
        a1.bar_label(bars, fmt="%.2f", fontsize=15, padding=4)
    a1.set_xticks(list(xs), [f"IoU ≥ {t}" for t in thr], fontsize=16)
    a1.set_ylim(0, 1)
    a1.tick_params(axis="y", labelsize=14)
    a1.set_ylabel("F1 against the hand annotation", fontsize=16)
    a1.set_title(
        f"Multiview beats the baseline at every matching threshold\n"
        f"({e['n_gt_in_region']} hand-drawn stones; reported F1 uses IoU ≥ "
        f"{e['iou_threshold']})",
        fontsize=18,
        loc="left",
    )
    a1.legend(fontsize=15, frameon=False)
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    rows = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Mean IoU (matched)", "mean_iou_matched"),
    ]
    labels = [r[0] for r in rows]
    yy = range(len(rows))
    for k, (meth, col) in enumerate((("ortho", "#9aa4ae"), ("multiview", ACCENT))):
        vals = [e[meth][r[1]] for r in rows]
        bars = a2.barh([y + (k - 0.5) * 0.36 for y in yy], vals, 0.36, color=col)
        a2.bar_label(bars, fmt="%.2f", fontsize=14, padding=4)
    a2.set_yticks(list(yy), labels, fontsize=15)
    a2.invert_yaxis()
    a2.set_xlim(0, 1)
    a2.tick_params(axis="x", labelsize=14)
    a2.set_title("At IoU ≥ 0.5: the gain is recall", fontsize=18, loc="left")
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
    fig.tight_layout(pad=3)
    out = []
    for ext in ("svg", "png"):
        p = OUT / f"07_results_segmentation.{ext}"
        fig.savefig(p, facecolor="white")
        out.append(p)
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / "outputs" / "highwall" / "metrics.json").read_text(encoding="utf-8"))
    made = [
        architecture(m),
        pipeline(m),
        multiview(m),
        registration(m),
        evaluation(m),
        wall_frame(m),
        *results_chart(m),
    ]
    for p in made:
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
