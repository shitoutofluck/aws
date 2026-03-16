"""Main entry point for the agentic orchestration system."""

import asyncio
import sys
from pathlib import Path

import structlog

from src.config import get_settings
from src.graph import create_workflow
from src.llms import LLMInitializationError, initialize_all_llms
from src.tools.browser import cleanup_browser


def setup_logging() -> None:
    """Configure structured logging."""
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.log_level == "DEBUG" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


async def run_task(task: str) -> dict:
    """
    Run a task through the orchestration system.

    Args:
        task: Task description from user

    Returns:
        dict: Final state after execution
    """
    logger = structlog.get_logger()

    try:
        logger.info("task_started", task=task)

        # Create workflow
        workflow = create_workflow()

        # Initial state
        initial_state = {
            "task": task,
            "plan": None,
            "current_step": 0,
            "results": [],
            "context": {},
            "error": None,
            "completed": False,
        }

        # Execute workflow
        logger.info("executing_workflow")
        final_state = await workflow.ainvoke(initial_state)

        logger.info("workflow_completed", completed=final_state.get("completed"))

        return final_state

    except Exception as e:
        logger.error("workflow_execution_failed", error=str(e), exc_info=True)
        raise


async def main_async() -> int:
    """
    Async main function.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger = structlog.get_logger()
    settings = get_settings()

    try:
        # Display banner
        print("\n" + "=" * 70)
        print("  Autonomous Multi-LLM Orchestration System")
        print("=" * 70)
        print(f"  Safety Mode: {settings.safety_mode.value.upper()}")
        print(f"  DeepSeek: {settings.deepseek_mode.value} mode")
        print("=" * 70 + "\n")

        # Get task from command line or prompt
        if len(sys.argv) > 1:
            task = " ".join(sys.argv[1:])
        else:
            print("Enter your task (or 'quit' to exit):")
            task = input("> ").strip()

            if task.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                return 0

        if not task:
            print("Error: No task provided")
            return 1

        logger.info("initializing_system", task=task)

        # Initialize LLMs
        print("\nInitializing LLM clients...")
        try:
            llms = initialize_all_llms()
            print(f"✓ Initialized {len(llms)} LLM(s): {', '.join(llms.keys())}")
        except LLMInitializationError as e:
            print(f"\n✗ LLM Initialization Failed:\n{str(e)}")
            return 1

        # Run task
        print(f"\nExecuting task: {task}\n")
        print("-" * 70)

        final_state = await run_task(task)

        print("-" * 70)

        # Display results
        if final_state.get("error"):
            print(f"\n✗ Task failed with error:\n{final_state['error']}\n")
            return 1

        results = final_state.get("results", [])
        if results:
            print("\n✓ Task completed successfully!\n")
            print("Results:")
            print("=" * 70)

            for i, result in enumerate(results, 1):
                if isinstance(result, dict):
                    output = result.get("output", result.get("summary", ""))
                    if output:
                        print(f"\n{output}")
                        print("-" * 70)
                else:
                    print(f"\n{result}")
                    print("-" * 70)

            print()

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up...")
        return 1

    except Exception as e:
        logger.error("main_failed", error=str(e), exc_info=True)
        print(f"\n✗ Unexpected error: {str(e)}\n")
        return 1

    finally:
        # Cleanup
        logger.info("cleaning_up")
        await cleanup_browser()


def main() -> int:
    """
    Main entry point.

    Returns:
        int: Exit code
    """
    # Setup logging
    setup_logging()

    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)

    # Run async main
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nExiting...")
        return 1


if __name__ == "__main__":
    sys.exit(main())
