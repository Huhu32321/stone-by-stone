"""Shared machinery for pipeline stages.

Every stage writes only inside ``outputs/<scene>/<NN_stage>/`` and records a
``stage.json`` there with inputs, parameters, tool versions, wall-clock time and metrics.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stonetwin import __version__

# Stage name -> output folder name. Order is pipeline order.
STAGE_DIRS: dict[str, str] = {
    "ingest": "00_ingest",
    "sfm": "01_sfm",
    "dense": "02_dense",
    "splat": "03_splat",
    "frame": "04_frame",
    "register": "05_register",
    "segment": "06_segment",
    "attributes": "07_attributes",
    "evaluate": "08_evaluate",
    "export-viewer": "09_export_viewer",
    "report": "10_report",
}


class StageError(RuntimeError):
    """A stage could not run or failed an acceptance check. Exits non-zero."""


class StageNotImplementedError(StageError):
    """The stage is still a stub."""


class AcceptanceError(StageError):
    """The stage ran but its results fail an acceptance check.

    ``run_stage`` still writes ``stage.json`` (status ``failed_acceptance``) so the
    metrics are available for diagnosis, then re-raises.
    """


@dataclass
class StageContext:
    """What a stage needs to run.

    Attributes:
        name: Stage name, a key of ``STAGE_DIRS``.
        scene: Scene name.
        cfg: Merged config (see ``stonetwin.config.load_config``).
        force: Recompute even if a previous ``stage.json`` exists.
    """

    name: str
    scene: str
    cfg: dict[str, Any]
    force: bool = False
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        """Absolute repository root."""
        return Path(self.cfg["_root"])

    @property
    def scene_dir(self) -> Path:
        """``outputs/<scene>/``."""
        return self.root / "outputs" / self.scene

    @property
    def out_dir(self) -> Path:
        """This stage's output folder, ``outputs/<scene>/<NN_stage>/``."""
        return self.scene_dir / STAGE_DIRS[self.name]

    def stage_dir(self, name: str) -> Path:
        """Output folder of another stage of the same scene."""
        return self.scene_dir / STAGE_DIRS[name]

    @property
    def stage_json(self) -> Path:
        return self.out_dir / "stage.json"

    def is_done(self) -> bool:
        """True if a completed ``stage.json`` exists and ``--force`` was not given."""
        if self.force or not self.stage_json.is_file():
            return False
        try:
            return json.loads(self.stage_json.read_text(encoding="utf-8")).get("status") == "ok"
        except json.JSONDecodeError:
            return False

    def write_stage_json(self, started: float, status: str = "ok") -> Path:
        """Write ``stage.json`` for this run."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": self.name,
            "scene": self.scene,
            "status": status,
            "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "wall_clock_s": round(time.perf_counter() - started, 3),
            "inputs": self.inputs,
            "params": self.params,
            "tools": {"stonetwin": __version__, "python": platform.python_version(), **self.tools},
            "metrics": self.metrics,
        }
        self.stage_json.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return self.stage_json


StageFn = Callable[[StageContext], None]


def run_stage(ctx: StageContext, fn: StageFn) -> bool:
    """Run ``fn`` under ``ctx`` unless already done.

    Returns:
        True if the stage ran, False if it was skipped because outputs exist.

    Raises:
        StageError: propagated from the stage, so the CLI can exit non-zero.
    """
    if ctx.is_done():
        return False
    started = time.perf_counter()
    try:
        fn(ctx)
    except AcceptanceError:
        ctx.write_stage_json(started, status="failed_acceptance")
        raise
    ctx.write_stage_json(started)
    return True
