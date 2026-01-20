"""Configuration management using Pydantic settings."""

from enum import Enum
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SafetyMode(str, Enum):
    """Safety mode for computer control operations."""

    STRICT = "strict"  # Read-only operations, confirmation required
    PERMISSIVE = "permissive"  # Allows write operations with validation


class DeepSeekMode(str, Enum):
    """DeepSeek deployment mode."""

    LOCAL = "local"  # Via Ollama
    API = "api"  # Via API (if available)


class Settings(BaseSettings):
    """Application settings with validation."""

    # API Keys
    anthropic_api_key: SecretStr
    xai_api_key: SecretStr
    deepseek_api_key: SecretStr | None = None

    # LLM Configuration
    deepseek_mode: DeepSeekMode = DeepSeekMode.LOCAL
    deepseek_model: str = "deepseek-r1:8b"
    claude_model: str = "claude-3-5-sonnet-20241022"
    grok_model: str = "grok-beta"

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"

    # xAI Configuration
    xai_base_url: str = "https://api.x.ai/v1"

    # Safety Configuration
    safety_mode: SafetyMode = SafetyMode.STRICT
    require_confirmation: bool = True
    action_timeout: int = 30  # seconds
    screenshot_logging: bool = True

    # Application Configuration
    log_level: str = "INFO"
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds

    # Computer Control Limits
    max_screen_width: int = 3840
    max_screen_height: int = 2160

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("action_timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is reasonable."""
        if v < 1 or v > 300:
            raise ValueError("Action timeout must be between 1 and 300 seconds")
        return v


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings instance (useful for testing)."""
    global _settings
    _settings = None
