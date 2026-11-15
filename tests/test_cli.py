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


def test_failing_stage_exits_nonzero_without_stage_json(monkeypatch):
    # A stage that raises StageError must exit 1 and must not write a stage.json.
    # Uses a stand-in for `report` so the real stage (which rewrites docs/) never runs.
    import stonetwin.cli as cli
    from stonetwin.stage import StageError

    def boom(ctx):
        raise StageError("deliberate failure for the test")

    monkeypatch.chdir(ROOT)
    monkeypatch.setitem(cli.STAGES, "report", boom)
    monkeypatch.setitem(STAGE_DIRS, "report", "zz_test_failing_stage")
    out = ROOT / "outputs" / "highwall" / "zz_test_failing_stage"
    result = runner.invoke(app, ["report", "--scene", "highwall", "--force"])
    try:
        assert result.exit_code == 1
        assert "deliberate failure" in result.output
        assert not (out / "stage.json").exists()
    finally:
        if out.is_dir() and not any(out.iterdir()):
            out.rmdir()


def test_unknown_scene_exits_2(monkeypatch):
    monkeypatch.chdir(ROOT)
    result = runner.invoke(app, ["ingest", "--scene", "nope"])
    assert result.exit_code == 2
