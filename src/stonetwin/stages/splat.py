"""Stage `splat`: stub. See docs/TECH_SPEC.md section 4."""

from __future__ import annotations

from stonetwin.stage import StageContext, StageNotImplementedError


def run(ctx: StageContext) -> None:
    """Run the stage for ``ctx.scene``. Not implemented yet."""
    ctx.params = dict(ctx.cfg.get("splat", {}))
    raise StageNotImplementedError(
        f"stage '{ctx.name}' is not implemented yet (config section 'splat': {ctx.params})"
    )
