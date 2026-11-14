"""Locating and running external command-line tools (COLMAP, LichtFeld Studio)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from stonetwin.stage import StageError

_ENV_VARS = {"colmap": "STONETWIN_COLMAP", "lichtfeld": "STONETWIN_LICHTFELD"}
_PATH_NAMES = {"colmap": ["colmap", "COLMAP.bat"], "lichtfeld": ["LichtFeld-Studio"]}


def find_tool(name: str, cfg: dict[str, Any]) -> Path:
    """Resolve the executable for ``name``.

    Order: ``cfg['tools'][name]``, then the ``STONETWIN_<NAME>`` environment variable,
    then ``PATH``. For COLMAP a ``COLMAP.bat`` launcher is swapped for ``bin/colmap.exe``
    so that exit codes and output are not hidden by the batch wrapper.
    """
    candidates: list[str] = []
    configured = (cfg.get("tools") or {}).get(name)
    if configured:
        candidates.append(str(configured))
    if os.environ.get(_ENV_VARS[name]):
        candidates.append(os.environ[_ENV_VARS[name]])
    for exe in _PATH_NAMES[name]:
        found = shutil.which(exe)
        if found:
            candidates.append(found)
    for cand in candidates:
        path = Path(cand)
        if name == "colmap" and path.suffix.lower() == ".bat":
            exe = path.parent / "bin" / "colmap.exe"
            path = exe if exe.is_file() else path
        if path.is_file():
            return path
    raise StageError(
        f"{name} not found. Set tools.{name} in configs/local.yaml or {_ENV_VARS[name]}."
    )


def run_logged(args: list[str | Path], log_path: Path, cwd: Path | None = None) -> None:
    """Run a command, appending its stdout and stderr to ``log_path``.

    Raises:
        StageError: if the command exits non-zero. The message names the log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(a) for a in args]
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise StageError(
            f"{Path(cmd[0]).name} {cmd[1] if len(cmd) > 1 else ''} exited with "
            f"0x{proc.returncode & 0xFFFFFFFF:08X}; see {log_path}"
        )


def colmap_version(colmap: Path) -> str:
    """Return the COLMAP version string, e.g. ``"4.2.1 (bd1fcf6)"``."""
    out = subprocess.run(
        [str(colmap), "version"], capture_output=True, text=True, check=False
    ).stdout
    m = re.search(r"COLMAP (\S+) \(Commit (\w+)", out)
    return f"{m.group(1)} ({m.group(2)})" if m else out.strip().splitlines()[0]
