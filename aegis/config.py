"""Configuration settings for Aegis using pydantic-settings."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Aegis application settings loaded from environment variables (prefix AEGIS_)."""

    model_config = SettingsConfigDict(
        env_prefix="AEGIS_",
        env_file=".env",
        extra="ignore",
    )

    provider: Optional[Literal["anthropic", "openai"]] = Field(
        default=None,
        description="LLM provider: 'anthropic' or 'openai'. Required for task execution.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Default LLM model name.",
    )
    judge_model: Optional[str] = Field(
        default=None,
        description="Judge LLM model name for verification.",
    )
    agent_step_limit: int = Field(
        default=8,
        description="Maximum execution steps per agent.",
    )
    tool_output_limit: int = Field(
        default=8000,
        description="Maximum characters for tool output before truncation.",
    )
    sandbox_timeout: float = Field(
        default=20.0,
        description="Execution timeout for python sandbox in seconds.",
    )
    dataset_max_bytes: int = Field(
        default=20 * 1024 * 1024,
        description="Maximum dataset upload size in bytes (20 MB).",
    )
    dataset_max_rows: int = Field(
        default=200_000,
        description="Maximum number of rows in dataset.",
    )
    dataset_max_cols: int = Field(
        default=100,
        description="Maximum number of columns in dataset.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name.",
    )
    embedding_dim: int = Field(
        default=384,
        description="Embedding vector dimensions.",
    )


_global_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Retrieve the cached global settings instance."""
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings()
    return _global_settings


def reset_settings(new_settings: Optional[Settings] = None) -> None:
    """Reset or override global settings (useful for tests)."""
    global _global_settings
    _global_settings = new_settings
