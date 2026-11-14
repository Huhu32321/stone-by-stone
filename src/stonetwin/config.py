"""Configuration loading.

The merged config is ``configs/default.yaml``, overridden by
``configs/scenes/<scene>.yaml``, overridden by ``configs/local.yaml`` if it exists.
``local.yaml`` is git-ignored and holds machine-specific values such as tool paths.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    """Raised when a config file is missing or malformed."""


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor of ``start`` (default: cwd) holding ``pyproject.toml``."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise ConfigError(f"Could not find the repository root above {here}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``base`` recursively updated by ``override``.

    Nested dicts are merged; any other value (including lists) is replaced.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return data


def load_config(scene: str, config_path: Path | None = None, root: Path | None = None) -> dict:
    """Load the merged config for ``scene``.

    Args:
        scene: Scene name; ``configs/scenes/<scene>.yaml`` must exist.
        config_path: Defaults file. Relative paths resolve against the repo root.
        root: Repository root. Found from the working directory if omitted.

    Returns:
        The merged config dict, with ``_root`` (absolute repo root, as str) added.
    """
    root = root or find_repo_root()
    config_path = config_path or Path("configs/default.yaml")
    if not config_path.is_absolute():
        config_path = root / config_path

    cfg = _read_yaml(config_path)
    scene_path = root / "configs" / "scenes" / f"{scene}.yaml"
    if not scene_path.is_file():
        known = sorted(p.stem for p in (root / "configs" / "scenes").glob("*.yaml"))
        raise ConfigError(f"Unknown scene '{scene}'. Known scenes: {', '.join(known)}")
    cfg = deep_merge(cfg, _read_yaml(scene_path))

    local_path = root / "configs" / "local.yaml"
    if local_path.is_file():
        cfg = deep_merge(cfg, _read_yaml(local_path))

    if cfg.get("scene") != scene:
        raise ConfigError(f"{scene_path} declares scene '{cfg.get('scene')}', expected '{scene}'")
    cfg["_root"] = str(root)
    return cfg
