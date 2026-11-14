"""Command-line interface: ``python -m stonetwin <stage> --scene <name>``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from stonetwin.config import ConfigError, load_config
from stonetwin.stage import STAGE_DIRS, StageContext, StageError, run_stage
from stonetwin.stages import HELP, STAGES

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Stone by Stone: per-stone digital twin pipeline. One sub-command per stage.",
)

SceneOpt = Annotated[str, typer.Option("--scene", "-s", help="Scene name, e.g. highwall.")]
ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Defaults file, merged with the scene file.")
]
ForceOpt = Annotated[bool, typer.Option("--force", help="Recompute even if outputs exist.")]


def _execute(stage: str, scene: str, config: Path, force: bool) -> None:
    try:
        cfg = load_config(scene, config)
    except ConfigError as exc:
        typer.secho(f"config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    ctx = StageContext(name=stage, scene=scene, cfg=cfg, force=force)
    try:
        ran = run_stage(ctx, STAGES[stage])
    except StageError as exc:
        typer.secho(f"[{stage}] failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc
    msg = "done" if ran else "up to date (use --force to recompute)"
    typer.echo(f"[{stage}] {scene}: {msg} -> {ctx.out_dir}")


def _make_command(stage: str):
    def command(
        scene: SceneOpt,
        config: ConfigOpt = Path("configs/default.yaml"),
        force: ForceOpt = False,
    ) -> None:
        _execute(stage, scene, config, force)

    command.__doc__ = HELP[stage]
    return command


for _stage in STAGE_DIRS:
    app.command(name=_stage, help=HELP[_stage])(_make_command(_stage))


@app.command(name="all")
def run_all(
    scene: SceneOpt,
    config: ConfigOpt = Path("configs/default.yaml"),
    force: ForceOpt = False,
) -> None:
    """Run every stage in pipeline order for one scene."""
    for stage in STAGE_DIRS:
        _execute(stage, scene, config, force)
