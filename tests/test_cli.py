from pathlib import Path

from typer.testing import CliRunner

from stonetwin.cli import app
from stonetwin.config import find_repo_root
from stonetwin.stage import STAGE_DIRS

runner = CliRunner()
ROOT = find_repo_root(Path(__file__).parent)


def test_help_lists_every_stage():
    result = runner.invoke(app, ["--help"], terminal_width=200)
    assert result.exit_code == 0
    for stage in [*STAGE_DIRS, "all"]:
        assert stage in result.output


def test_stub_stage_exits_nonzero_without_stage_json(monkeypatch):
    # 'report' is still a stub; it must fail loudly and not write a fake stage.json.
    monkeypatch.chdir(ROOT)
    out = ROOT / "outputs" / "highwall" / STAGE_DIRS["report"] / "stage.json"
    existed = out.exists()
    result = runner.invoke(app, ["report", "--scene", "highwall"])
    assert result.exit_code == 1
    assert "not implemented" in result.output
    assert out.exists() == existed


def test_unknown_scene_exits_2(monkeypatch):
    monkeypatch.chdir(ROOT)
    result = runner.invoke(app, ["ingest", "--scene", "nope"])
    assert result.exit_code == 2
