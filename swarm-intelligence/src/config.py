"""Configuration management for swarm intelligence simulation."""

from enum import Enum

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepSeekMode(str, Enum):
    """DeepSeek deployment mode."""

    LOCAL = "local"  # Via Ollama
    API = "api"  # Via API


class SimulationMode(str, Enum):
    """Simulation execution mode."""

    LLM = "llm"          # Full LLM-powered agent decisions (richer, slower)
    RULE_BASED = "rule"   # Rule-based decisions (fast, cost-free)
    HYBRID = "hybrid"     # LLM for key agents, rules for the rest


class Settings(BaseSettings):
    """Application settings with validation."""

    # API Keys (optional — system works with Ollama alone)
    anthropic_api_key: SecretStr | None = None
    xai_api_key: SecretStr | None = None
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

    # Simulation Configuration
    simulation_mode: SimulationMode = SimulationMode.HYBRID
    max_agents: int = 20          # Max agents to generate from graph
    min_agents: int = 5           # Min agents (synthetic if graph is sparse)
    max_rounds: int = 10          # Simulation rounds
    use_llm_for_bios: bool = True # Use LLM for rich agent biographies
    chunk_size: int = 500         # Text chunking for GraphRAG
    chunk_overlap: int = 50       # Chunk overlap

    # Memory Configuration
    memory_storage_dir: str = "data/memory"
    max_memory_entries: int = 100

    # Application Configuration
    log_level: str = "INFO"
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    data_dir: str = "data"     # Base data directory

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("max_rounds")
    @classmethod
    def validate_rounds(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("max_rounds must be between 1 and 100")
        return v

    @field_validator("max_agents")
    @classmethod
    def validate_agents(cls, v: int) -> int:
        if v < 1 or v > 500:
            raise ValueError("max_agents must be between 1 and 500")
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
