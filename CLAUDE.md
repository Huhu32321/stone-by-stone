# Instructions for Claude Code

You are the engineer on this project. The product decisions are already made; your job
is to build what the documents specify, on this machine, and to report honestly.

## Read first, in this order

1. `docs/PRD.md` — what is being built and why, priorities, risks.
2. `docs/TECH_SPEC.md` — stage contracts, outputs, acceptance checks.
3. `docs/TASKS.md` — the build order. This is your to-do list.
4. `docs/DATA.md` — what the photos are and what is wrong with them.
5. `docs/DECISIONS.md` — choices already made, and where you log new ones.
6. `docs/internal/CONTEXT.md` — private background on who the result is for. Never
   quote it in anything that is committed.

## How to work

- Follow `docs/TASKS.md` top to bottom. Tick boxes in that file as you go.
- Stop at every **GATE**, write the gate report it asks for, and wait for the author.
  Do not start the next day's milestones on your own.
- When the spec says "choose in M0" or leaves a parameter open, choose, then add an
  entry to `docs/DECISIONS.md` with the reason. Do the same whenever you deviate from
  the spec.
- If an acceptance check fails twice, stop and report what you tried. Do not loosen a
  threshold to make it pass.
- Prefer a working end-to-end path over a perfect stage. The orthophoto baseline
  exists so that there is always something to ship.

## Hard rules

1. **Never modify, move or delete anything under `data/raw/`.**
2. **Never write a metric by hand.** Numbers in `README.md` and `docs/RESULTS.md` come
   from `python -m stonetwin report` and nowhere else. If a result is missing, leave
   the placeholder.
3. **Never describe the condition proxies as deterioration classification.** Use the
   field names and wording in `docs/TECH_SPEC.md` section 4.7.
4. **Nothing private leaves the machine.** No raw photos, no EXIF GPS or serial
   numbers, no identifiable people, nothing from `docs/internal/` in git or in the
   viewer build. Check `git status` before every commit.
5. **Do not push, publish or deploy.** Prepare the workflow; the author presses the
   button.
6. **Report limits plainly.** A result below target is reported as it is, with the
   likely cause.

## Conventions

- Python 3.11+, package in `src/stonetwin/`, CLI via `python -m stonetwin`.
- Type hints on public functions. Docstrings state units and coordinate frames.
- Coordinate frames are named in variables: `p_world`, `p_wall`, `T_wall_from_world`.
- Lengths are metres internally once the scene is metric. Convert only for display.
- Config values come from YAML, not from constants scattered in code.
- Paths are built with `pathlib` and work on Windows. No hard-coded drive letters.
- Commits are small and use Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`).
- `ruff check` and `pytest` pass before each commit.

## Repository map

```
configs/        default.yaml and one file per scene
data/           raw photos (ignored), hand-entered measurements, annotations
docs/           requirements, spec, plan, dataset card, decisions, results
outputs/        everything the pipeline generates (ignored)
scripts/        one-off helpers that are not pipeline stages
src/stonetwin/  the pipeline
tests/          unit tests
viewer/         static web viewer
```

## What the author provides, and when

- Reference distances for scale: `data/meta/highwall/scale_refs.json`. May arrive
  late. Until then, run non-metric and keep the flag.
- Ground-truth stone polygons: after you produce the orthophoto in M4. Tell the author
  the file to annotate, the tool to use, and where to save the result.

## First session

Run M0 from `docs/TASKS.md`. Before installing anything, report what you found on the
machine and what you intend to install.
