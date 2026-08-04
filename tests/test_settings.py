from pathlib import Path

import pytest

from ke.settings import Settings


def test_default_yaml_loads_nested_pipeline_settings() -> None:
    settings = Settings.load()

    assert settings.extract_model == "openai:gpt-5.6-terra"
    assert settings.resolve_model == "openai:gpt-5.6-terra"
    assert settings.synthesis_model == "openai:gpt-5.6-sol"
    assert settings.critic_model == "openai:gpt-5.6-sol"
    assert settings.chunking.max_tokens == 1800
    assert settings.ontology.strict_grounding is True
    assert settings.synthesis.critic_repairs == 1
    assert settings.runtime.require_citations is True


def test_environment_overrides_workspace(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "artifacts"
    monkeypatch.setenv("KE_WORKSPACE", str(workspace))

    settings = Settings.load()

    assert settings.workspace == Path(workspace)


def test_dotenv_overrides_custom_yaml(tmp_path, monkeypatch) -> None:
    config = tmp_path / "custom.yaml"
    config.write_text("extract_model: yaml:model\n", encoding="utf-8")
    (tmp_path / ".env").write_text("KE_EXTRACT_MODEL=dotenv:model\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = Settings.load(config)

    assert settings.extract_model == "dotenv:model"


def test_dotenv_loads_openai_key_as_masked_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-openai-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = Settings.load()

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-openai-key"
    assert "test-openai-key" not in repr(settings)


def test_explicit_missing_settings_file_is_an_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="settings file does not exist"):
        Settings.load(tmp_path / "missing.yaml")
