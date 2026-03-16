"""LangGraph workflow for swarm intelligence simulation.

Replaces the original linear task-execution graph with a MiroFish-style
simulation pipeline: Document -> Graph -> Agents -> Simulation -> Report.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from src.agents.agent_generator import AgentGenerator
from src.agents.agent_profile import AgentProfile
from src.agents.report_agent import ReportAgent
from src.config import SimulationMode, get_settings
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.ontology import OntologyGenerator
from src.llms import get_best_available_llm, get_claude_llm, get_deepseek_llm
from src.simulation.engine import RoundSummary, SimulationEngine
from src.simulation.events import EventQueue
from src.simulation.memory import MemoryManager

logger = structlog.get_logger()


class SimulationState(TypedDict):
    """State flowing through the simulation pipeline."""

    # Input
    document: str                                     # Source document text
    topic: str                                        # Topic/prediction query
    events_to_inject: list[str]                       # Events for God's Eye View

    # Pipeline stages
    knowledge_graph: KnowledgeGraph | None            # Built knowledge graph
    agents: list[AgentProfile]                        # Generated agents
    simulation_engine: SimulationEngine | None        # Simulation instance
    round_summaries: list[RoundSummary]               # Round-by-round results

    # Output
    report: dict[str, Any] | None                     # Final analysis report
    status_log: Annotated[list[str], operator.add]    # Progress log
    error: str | None
    completed: bool


async def build_knowledge_graph_node(state: SimulationState) -> SimulationState:
    """Stage 1: Build knowledge graph from document using local GraphRAG."""
    logger.info("stage_1_building_knowledge_graph")
    settings = get_settings()

    try:
        llm = get_best_available_llm()

        kg = KnowledgeGraph()
        info = await kg.build_from_document(
            text=state["document"],
            llm=llm,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        logger.info("knowledge_graph_built", nodes=info.node_count, edges=info.edge_count)

        return {
            **state,
            "knowledge_graph": kg,
            "status_log": [
                f"Knowledge graph built: {info.node_count} entities, "
                f"{info.edge_count} relationships, types: {info.entity_types}"
            ],
        }

    except Exception as e:
        logger.error("knowledge_graph_build_failed", error=str(e))
        return {
            **state,
            "error": f"Knowledge graph construction failed: {e}",
            "completed": True,
        }


async def generate_agents_node(state: SimulationState) -> SimulationState:
    """Stage 2: Generate autonomous agents from knowledge graph entities."""
    logger.info("stage_2_generating_agents")
    settings = get_settings()

    if not state.get("knowledge_graph"):
        return {**state, "error": "No knowledge graph available", "completed": True}

    try:
        llm = get_best_available_llm()
        generator = AgentGenerator(state["knowledge_graph"])

        agents = await generator.generate_agents(
            llm=llm,
            max_agents=settings.max_agents,
            min_agents=settings.min_agents,
            use_llm_for_bios=settings.use_llm_for_bios,
        )

        archetypes = {}
        for a in agents:
            archetypes[a.archetype] = archetypes.get(a.archetype, 0) + 1

        logger.info("agents_generated", count=len(agents), archetypes=archetypes)

        return {
            **state,
            "agents": agents,
            "status_log": [
                f"Generated {len(agents)} agents with archetypes: {archetypes}"
            ],
        }

    except Exception as e:
        logger.error("agent_generation_failed", error=str(e))
        return {**state, "error": f"Agent generation failed: {e}", "completed": True}


async def run_simulation_node(state: SimulationState) -> SimulationState:
    """Stage 3: Run the OASIS-style social simulation."""
    logger.info("stage_3_running_simulation")
    settings = get_settings()

    if not state.get("agents"):
        return {**state, "error": "No agents available", "completed": True}

    try:
        sim_llm = None
        if settings.simulation_mode in (SimulationMode.LLM, SimulationMode.HYBRID):
            sim_llm = get_best_available_llm()

        engine = SimulationEngine(
            agents=state["agents"],
            topic=state["topic"],
            max_rounds=settings.max_rounds,
            llm=sim_llm,
        )

        for event in state.get("events_to_inject", []):
            engine.inject_event(event)

        summaries = await engine.run()

        logger.info("simulation_completed", rounds=len(summaries))

        return {
            **state,
            "simulation_engine": engine,
            "round_summaries": summaries,
            "status_log": [
                f"Simulation completed: {len(summaries)} rounds, "
                f"{len(engine.all_actions)} total actions"
            ],
        }

    except Exception as e:
        logger.error("simulation_failed", error=str(e))
        return {**state, "error": f"Simulation failed: {e}", "completed": True}


async def generate_report_node(state: SimulationState) -> SimulationState:
    """Stage 4: Generate analysis report from simulation results."""
    logger.info("stage_4_generating_report")

    if not state.get("simulation_engine"):
        return {**state, "error": "No simulation data available", "completed": True}

    try:
        llm = get_claude_llm() or get_best_available_llm()

        report_agent = ReportAgent(llm=llm)
        report = await report_agent.generate_report(state["simulation_engine"])

        logger.info("report_generated")

        return {
            **state,
            "report": report,
            "status_log": ["Analysis report generated"],
            "completed": True,
        }

    except Exception as e:
        logger.error("report_generation_failed", error=str(e))
        return {
            **state,
            "error": f"Report generation failed: {e}",
            "completed": True,
        }


def route_after_stage(state: SimulationState) -> Literal["next", END]:
    """Route to next stage or end on error."""
    if state.get("error") or state.get("completed"):
        return END
    return "next"


def create_workflow() -> Any:
    """Create the LangGraph simulation pipeline.

    Pipeline: Document -> Knowledge Graph -> Agent Generation -> Simulation -> Report
    """
    logger.info("creating_simulation_workflow")

    workflow = StateGraph(SimulationState)

    workflow.add_node("build_graph", build_knowledge_graph_node)
    workflow.add_node("generate_agents", generate_agents_node)
    workflow.add_node("run_simulation", run_simulation_node)
    workflow.add_node("generate_report", generate_report_node)

    workflow.add_edge(START, "build_graph")

    workflow.add_conditional_edges(
        "build_graph",
        route_after_stage,
        {"next": "generate_agents", END: END},
    )

    workflow.add_conditional_edges(
        "generate_agents",
        route_after_stage,
        {"next": "run_simulation", END: END},
    )

    workflow.add_conditional_edges(
        "run_simulation",
        route_after_stage,
        {"next": "generate_report", END: END},
    )

    workflow.add_edge("generate_report", END)

    logger.info("simulation_workflow_created")
    return workflow.compile()
