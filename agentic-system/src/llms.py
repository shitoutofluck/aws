"""LLM client initialization and management."""

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import DeepSeekMode, get_settings

logger = structlog.get_logger()


class LLMInitializationError(Exception):
    """Raised when LLM initialization fails."""

    pass


def get_claude_llm() -> ChatAnthropic:
    """
    Initialize Claude 3.5 Sonnet client.

    Returns:
        ChatAnthropic: Configured Claude client for planning/orchestration.

    Raises:
        LLMInitializationError: If initialization fails.
    """
    settings = get_settings()
    try:
        logger.info("initializing_claude", model=settings.claude_model)
        llm = ChatAnthropic(
            model=settings.claude_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            temperature=0.7,
            max_tokens=4096,
            timeout=60.0,
        )
        logger.info("claude_initialized_successfully")
        return llm
    except Exception as e:
        logger.error("claude_initialization_failed", error=str(e))
        raise LLMInitializationError(f"Failed to initialize Claude: {e}") from e


def get_grok_llm() -> ChatOpenAI:
    """
    Initialize Grok client via xAI API (OpenAI-compatible).

    Returns:
        ChatOpenAI: Configured Grok client for computer control.

    Raises:
        LLMInitializationError: If initialization fails.
    """
    settings = get_settings()
    try:
        logger.info("initializing_grok", model=settings.grok_model, base_url=settings.xai_base_url)
        llm = ChatOpenAI(
            model=settings.grok_model,
            api_key=settings.xai_api_key.get_secret_value(),
            base_url=settings.xai_base_url,
            temperature=0.5,
            max_tokens=2048,
            timeout=60.0,
        )
        logger.info("grok_initialized_successfully")
        return llm
    except Exception as e:
        logger.error("grok_initialization_failed", error=str(e))
        raise LLMInitializationError(f"Failed to initialize Grok: {e}") from e


def get_deepseek_llm() -> ChatOllama | ChatOpenAI:
    """
    Initialize DeepSeek client (local via Ollama or API).

    Returns:
        ChatOllama | ChatOpenAI: Configured DeepSeek client for browser automation.

    Raises:
        LLMInitializationError: If initialization fails.
    """
    settings = get_settings()

    if settings.deepseek_mode == DeepSeekMode.LOCAL:
        try:
            logger.info(
                "initializing_deepseek_local",
                model=settings.deepseek_model,
                base_url=settings.ollama_base_url,
            )
            llm = ChatOllama(
                model=settings.deepseek_model,
                base_url=settings.ollama_base_url,
                temperature=0.3,
                num_ctx=4096,
            )
            logger.info("deepseek_local_initialized_successfully")
            return llm
        except Exception as e:
            logger.error("deepseek_local_initialization_failed", error=str(e))
            error_msg = (
                f"Failed to initialize local DeepSeek via Ollama: {e}\n\n"
                "Make sure Ollama is installed and running:\n"
                "1. Install Ollama: https://ollama.ai\n"
                f"2. Pull the model: ollama pull {settings.deepseek_model}\n"
                "3. Verify Ollama is running: ollama list"
            )
            raise LLMInitializationError(error_msg) from e
    else:
        # API mode (if DeepSeek provides API access)
        if not settings.deepseek_api_key:
            raise LLMInitializationError(
                "DeepSeek API key required when using API mode. Set DEEPSEEK_API_KEY environment variable."
            )
        try:
            logger.info("initializing_deepseek_api", model=settings.deepseek_model)
            # This is a placeholder - adjust base_url when DeepSeek API is available
            llm = ChatOpenAI(
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key.get_secret_value(),
                base_url="https://api.deepseek.com/v1",  # Placeholder
                temperature=0.3,
                max_tokens=2048,
            )
            logger.info("deepseek_api_initialized_successfully")
            return llm
        except Exception as e:
            logger.error("deepseek_api_initialization_failed", error=str(e))
            raise LLMInitializationError(f"Failed to initialize DeepSeek API: {e}") from e


def initialize_all_llms() -> dict[str, ChatAnthropic | ChatOpenAI | ChatOllama]:
    """
    Initialize all LLM clients.

    Returns:
        dict: Dictionary mapping LLM names to initialized clients.

    Raises:
        LLMInitializationError: If any LLM initialization fails.
    """
    logger.info("initializing_all_llms")

    llms = {}

    try:
        llms["claude"] = get_claude_llm()
        llms["grok"] = get_grok_llm()
        llms["deepseek"] = get_deepseek_llm()
    except LLMInitializationError:
        logger.error("llm_initialization_failed", initialized=list(llms.keys()))
        raise

    logger.info("all_llms_initialized_successfully", llms=list(llms.keys()))
    return llms
