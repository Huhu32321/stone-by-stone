from pathlib import Path

import pytest

from stonetwin.config import ConfigError, deep_merge, find_repo_root, load_config

ROOT = find_repo_root(Path(__file__).parent)


def test_deep_merge_nested_and_replace():
    base = {"a": 1, "b": {"c": 2, "d": [1, 2]}}
    over = {"b": {"d": [3], "e": 4}, "f": 5}
    merged = deep_merge(base, over)
    assert merged == {"a": 1, "b": {"c": 2, "d": [3], "e": 4}, "f": 5}
    assert base == {"a": 1, "b": {"c": 2, "d": [1, 2]}}  # input untouched


@pytest.mark.parametrize("scene", ["highwall", "entrance"])
def test_load_scene_configs(scene):
    cfg = load_config(scene, root=ROOT)
    assert cfg["scene"] == scene
    assert cfg["sfm"]["accept"]["min_registered_fraction"] == pytest.approx(0.90)
    assert Path(cfg["_root"]) == ROOT


def test_scene_overrides_defaults():
    assert load_config("entrance", root=ROOT)["segment"]["methods"] == ["multiview"]
    assert load_config("highwall", root=ROOT)["segment"]["methods"] == ["ortho", "multiview"]


def test_unknown_scene_raises():
    with pytest.raises(ConfigError, match="Unknown scene"):
        load_config("nope", root=ROOT)
