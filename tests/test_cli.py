import json
from pathlib import Path

from typer.testing import CliRunner

from ke.cli import app

runner = CliRunner()


def test_cli_help_lists_all_pipeline_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "pdf-to-markdown",
        "ingest",
        "extract",
        "synthesize",
        "build",
        "validate",
        "ask",
    ):
        assert command in result.stdout


def test_synthesize_help_exposes_force_cache_bypass() -> None:
    result = runner.invoke(app, ["synthesize", "--help"])

    assert result.exit_code == 0
    assert "--force" in result.stdout


def test_offline_build_runs_all_three_pipelines(tmp_path: Path) -> None:
    fixtures = Path(__file__).parent / "fixtures"
    workspace = tmp_path / "workspace"
    objective = "Compare coordination and delegation while preserving source distinctions"

    result = runner.invoke(
        app,
        [
            "build",
            str(fixtures / "coordination.md"),
            str(fixtures / "autonomy.md"),
            "--name",
            "fixture-reasoner",
            "--objective",
            objective,
            "--offline",
            "--workspace",
            str(workspace),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["review_approved"] is True
    assert len(summary["ontology_ids"]) == 2

    synthesis_dir = Path(summary["output_dir"])
    assert {
        "alignment.yaml",
        "conflicts.yaml",
        "integrated-ontology.jsonld",
        "integrated-ontology.trig",
        "agent-frame.yaml",
        "stance-ledger.json",
        "reasoning-policy.md",
        "system-prompt.md",
        "tool-manifest.json",
        "review.json",
    } <= {path.name for path in synthesis_dir.iterdir()}

    prompt = (synthesis_dir / "system-prompt.md").read_text(encoding="utf-8")
    assert objective in prompt
    assert "instrumental" in prompt.casefold()
    assert "citation" in prompt.casefold() or "cite" in prompt.casefold()
    assert "source claims" in prompt.casefold()
    assert "synthesis" in prompt.casefold()
    assert "uncertainty" in prompt.casefold()
