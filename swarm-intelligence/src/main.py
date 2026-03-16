"""Main entry point for the swarm intelligence simulation system."""

import asyncio
import json
import os
import sys
from pathlib import Path

import structlog

from src.config import get_settings
from src.graph import create_workflow
from src.llms import LLMInitializationError, initialize_all_llms


def setup_logging() -> None:
    """Configure structured logging."""
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def load_document(path_or_text: str) -> str:
    """Load document from file path or use as raw text."""
    if os.path.isfile(path_or_text):
        with open(path_or_text, "r", encoding="utf-8") as f:
            return f.read()
    return path_or_text


async def run_simulation(
    document: str,
    topic: str,
    events: list[str] | None = None,
) -> dict:
    """Run a swarm intelligence simulation.

    Args:
        document: Source document text or file path.
        topic: Topic/question for the simulation.
        events: Optional events to inject (God's Eye View).

    Returns:
        Final simulation state.
    """
    logger = structlog.get_logger()

    try:
        logger.info("simulation_started", topic=topic, doc_length=len(document))

        workflow = create_workflow()

        initial_state = {
            "document": document,
            "topic": topic,
            "events_to_inject": events or [],
            "knowledge_graph": None,
            "agents": [],
            "simulation_engine": None,
            "round_summaries": [],
            "report": None,
            "status_log": [],
            "error": None,
            "completed": False,
        }

        final_state = await workflow.ainvoke(initial_state)
        return final_state

    except Exception as e:
        logger.error("simulation_failed", error=str(e), exc_info=True)
        raise


async def main_async() -> int:
    """Async main function."""
    logger = structlog.get_logger()
    settings = get_settings()

    try:
        # Banner
        print("\n" + "=" * 70)
        print("  Swarm Intelligence Simulation Engine")
        print("  Inspired by MiroFish - Digital Society Simulator")
        print("=" * 70)
        print(f"  Simulation Mode: {settings.simulation_mode.value}")
        print(f"  Max Agents: {settings.max_agents}")
        print(f"  Max Rounds: {settings.max_rounds}")
        print(f"  LLM Bios:   {settings.use_llm_for_bios}")
        print("=" * 70 + "\n")

        # Initialize LLMs
        print("Initializing LLM clients...")
        try:
            llms = initialize_all_llms()
            print(f"  Available LLMs: {', '.join(llms.keys())}")
        except LLMInitializationError as e:
            print(f"\n  LLM Initialization Failed:\n{e}")
            return 1

        # Parse arguments
        import argparse

        parser = argparse.ArgumentParser(description="Swarm Intelligence Simulation")
        parser.add_argument("document", nargs="?", help="Document file path or text")
        parser.add_argument("--topic", "-t", help="Simulation topic/question")
        parser.add_argument(
            "--event", "-e", action="append", default=[],
            help="Event to inject (can be repeated)"
        )
        parser.add_argument("--rounds", "-r", type=int, help="Override max rounds")
        parser.add_argument("--agents", "-a", type=int, help="Override max agents")

        args = parser.parse_args()

        # Get document
        if args.document:
            document = load_document(args.document)
        else:
            print("Enter document text (or file path). Press Ctrl+D when done:\n")
            try:
                document = sys.stdin.read().strip()
            except EOFError:
                document = ""

            if not document:
                print("Error: No document provided")
                return 1

        # Get topic
        topic = args.topic
        if not topic:
            print("\nEnter simulation topic/prediction question:")
            topic = input("> ").strip()

        if not topic:
            topic = "General analysis of the document contents"

        # Override settings if specified
        if args.rounds:
            settings.max_rounds = args.rounds
        if args.agents:
            settings.max_agents = args.agents

        # Run simulation
        print(f"\nDocument: {len(document)} characters")
        print(f"Topic: {topic}")
        if args.event:
            print(f"Events to inject: {args.event}")
        print("\n" + "-" * 70)
        print("Starting simulation pipeline...")
        print("-" * 70 + "\n")

        final_state = await run_simulation(
            document=document,
            topic=topic,
            events=args.event,
        )

        print("\n" + "-" * 70)

        # Display results
        if final_state.get("error"):
            print(f"\n  Simulation failed: {final_state['error']}")
            return 1

        # Status log
        status_log = final_state.get("status_log", [])
        if status_log:
            print("\nPipeline Progress:")
            for entry in status_log:
                print(f"  {entry}")

        # Report
        report = final_state.get("report")
        if report:
            print("\n" + "=" * 70)
            print("  SIMULATION REPORT")
            print("=" * 70)
            print(report.get("report_text", "No report generated."))
            print("=" * 70)

            # Save report
            data_dir = Path(settings.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

            report_path = data_dir / "latest_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                # Serialize report (skip non-serializable objects)
                serializable = {
                    "report_text": report.get("report_text"),
                    "analytics": report.get("analytics"),
                    "generated_at": report.get("generated_at"),
                }
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            print(f"\nReport saved to: {report_path}")

        # Save opinion trajectories
        engine = final_state.get("simulation_engine")
        if engine:
            traj_path = Path(settings.data_dir) / "opinion_trajectories.json"
            with open(traj_path, "w", encoding="utf-8") as f:
                json.dump(engine.get_opinion_trajectory(), f, indent=2)
            print(f"Opinion trajectories saved to: {traj_path}")

        print()
        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 1

    except Exception as e:
        logger.error("main_failed", error=str(e), exc_info=True)
        print(f"\nUnexpected error: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    setup_logging()
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nExiting...")
        return 1


if __name__ == "__main__":
    sys.exit(main())
