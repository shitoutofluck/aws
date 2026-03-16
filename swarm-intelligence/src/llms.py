"""LLM client initialization and management.

Supports three tiers:
1. Full mode: Claude + Grok + DeepSeek (all APIs configured)
2. Partial mode: Some APIs + local DeepSeek
3. Local-only mode: DeepSeek via Ollama (fully offline, cost-free)
"""

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.config import DeepSeekMode, get_settings

logger = structlog.get_logger()


class LLMInitializationError(Exception):
    """Raised when LLM initialization fails."""

    pass


def get_claude_llm() -> BaseChatModel | None:
    """Initialize Claude client. Returns None if API key not configured."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        logger.info("claude_skipped_no_api_key")
        return None

    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            temperature=0.7,
            max_tokens=4096,
            timeout=60.0,
        )
        logger.info("claude_initialized", model=settings.claude_model)
        return llm
    except Exception as e:
        logger.warning("claude_initialization_failed", error=str(e))
        return None


def get_grok_llm() -> BaseChatModel | None:
    """Initialize Grok client via xAI API. Returns None if API key not configured."""
    settings = get_settings()
    if not settings.xai_api_key:
        logger.info("grok_skipped_no_api_key")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.grok_model,
            api_key=settings.xai_api_key.get_secret_value(),
            base_url=settings.xai_base_url,
            temperature=0.5,
            max_tokens=2048,
            timeout=60.0,
        )
        logger.info("grok_initialized", model=settings.grok_model)
        return llm
    except Exception as e:
        logger.warning("grok_initialization_failed", error=str(e))
        return None


def get_deepseek_llm() -> BaseChatModel:
    """Initialize DeepSeek client (always available via Ollama).

    Raises:
        LLMInitializationError: If initialization fails.
    """
    settings = get_settings()

    if settings.deepseek_mode == DeepSeekMode.LOCAL:
        try:
            llm = ChatOllama(
                model=settings.deepseek_model,
                base_url=settings.ollama_base_url,
                temperature=0.3,
                num_ctx=4096,
            )
            logger.info("deepseek_local_initialized", model=settings.deepseek_model)
            return llm
        except Exception as e:
            error_msg = (
                f"Failed to initialize local DeepSeek via Ollama: {e}\n\n"
                "Make sure Ollama is installed and running:\n"
                "1. Install Ollama: https://ollama.ai\n"
                f"2. Pull the model: ollama pull {settings.deepseek_model}\n"
                "3. Verify Ollama is running: ollama list"
            )
            raise LLMInitializationError(error_msg) from e
    else:
        if not settings.deepseek_api_key:
            raise LLMInitializationError("DEEPSEEK_API_KEY required for API mode.")

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url="https://api.deepseek.com/v1",
            temperature=0.3,
            max_tokens=2048,
        )
        logger.info("deepseek_api_initialized", model=settings.deepseek_model)
        return llm


def get_best_available_llm() -> BaseChatModel:
    """Get the best available LLM in priority order: Claude > Grok > DeepSeek."""
    claude = get_claude_llm()
    if claude:
        return claude

    grok = get_grok_llm()
    if grok:
        return grok

    return get_deepseek_llm()


def initialize_all_llms() -> dict[str, BaseChatModel]:
    """Initialize all available LLM clients.

    At minimum, 'deepseek' will always be present (via Ollama).
    Claude and Grok are optional and depend on API keys.
    """
    llms: dict[str, BaseChatModel] = {}

    claude = get_claude_llm()
    if claude:
        llms["claude"] = claude

    grok = get_grok_llm()
    if grok:
        llms["grok"] = grok

    llms["deepseek"] = get_deepseek_llm()

    logger.info("llms_initialized", available=list(llms.keys()))
    return llms
