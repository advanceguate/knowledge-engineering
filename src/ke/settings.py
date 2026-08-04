"""Configuration loaded from YAML with environment overrides for secrets/models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource

from ke.resource_paths import resource_path


class ConfigSection(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConversionSettings(ConfigSection):
    backend: Literal["docling"] = "docling"
    ocr: str = "auto"
    table_structure: str = "accurate"
    export_markdown: bool = True
    export_docling_json: bool = True
    export_images: bool = True


class ChunkingSettings(ConfigSection):
    strategy: str = "docling_hybrid"
    max_tokens: int = Field(default=1800, ge=128)
    overlap_tokens: int = Field(default=150, ge=0)
    preserve_headings: bool = True
    preserve_tables: bool = True


class EntityResolutionSettings(ConfigSection):
    lexical_threshold: int = Field(default=90, ge=0, le=100)
    use_embeddings: bool = False
    llm_adjudicate_ambiguous_pairs: bool = True


class SalienceBudget(ConfigSection):
    terms: int = Field(default=250, ge=1)
    relations: int = Field(default=400, ge=1)
    axioms: int = Field(default=200, ge=1)
    mental_models: int = Field(default=60, ge=1)


class OntologySettings(ConfigSection):
    strict_grounding: bool = True
    minimum_confidence: float = Field(default=0.55, ge=0, le=1)
    reducer_batch_size: int = Field(default=8, ge=2)
    entity_resolution: EntityResolutionSettings = Field(default_factory=EntityResolutionSettings)
    salience_budget: SalienceBudget = Field(default_factory=SalienceBudget)


class SynthesisSettings(ConfigSection):
    objective_required: bool = True
    preserve_source_named_graphs: bool = True
    preserve_conflicts: bool = True
    critic_repairs: int = Field(default=1, ge=0, le=1)


class RuntimeSettings(ConfigSection):
    require_citations: bool = True
    expose_hidden_deliberation: bool = False


class Settings(BaseSettings):
    """Application settings.

    YAML controls reproducible pipeline behavior. Provider model identifiers and
    the workspace may be overridden with the ``KE_`` environment variables from
    ``.env.example``.
    """

    model_config = SettingsConfigDict(
        env_prefix="KE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    workspace: Path = Path("workspace")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    extract_model: str = "openai:gpt-5.6-terra"
    resolve_model: str = "openai:gpt-5.6-terra"
    synthesis_model: str = "openai:gpt-5.6-sol"
    critic_model: str = "openai:gpt-5.6-sol"
    llm_concurrency: int = Field(default=6, ge=1)

    conversion: ConversionSettings = Field(default_factory=ConversionSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    ontology: OntologySettings = Field(default_factory=OntologySettings)
    synthesis: SynthesisSettings = Field(default_factory=SynthesisSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        config_path = Path(path) if path is not None else resource_path("config", "default.yaml")
        if path is not None and not config_path.is_file():
            raise FileNotFoundError(f"settings file does not exist: {config_path}")
        payload: dict[str, object] = {}
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if loaded:
                if not isinstance(loaded, dict):
                    raise ValueError(f"settings file must contain a mapping: {config_path}")
                payload = loaded

        # BaseSettings gives constructor values priority over environment values,
        # so explicitly overlay fields present in the process environment or the
        # configured dotenv file after loading the reproducible YAML sections.
        env_fields = {
            "workspace": "KE_WORKSPACE",
            "extract_model": "KE_EXTRACT_MODEL",
            "resolve_model": "KE_RESOLVE_MODEL",
            "synthesis_model": "KE_SYNTHESIS_MODEL",
            "critic_model": "KE_CRITIC_MODEL",
            "llm_concurrency": "KE_LLM_CONCURRENCY",
        }
        dotenv_values = DotEnvSettingsSource(
            cls,
            env_file=cls.model_config.get("env_file"),
            env_file_encoding=cls.model_config.get("env_file_encoding"),
        )()
        environment = cls()
        for field, variable in env_fields.items():
            if variable in os.environ or field in dotenv_values:
                payload[field] = getattr(environment, field)
        return cls(**payload)


def load_settings(path: str | Path | None = None) -> Settings:
    return Settings.load(path)
