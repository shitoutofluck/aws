"""LangGraph workflow for multi-agent orchestration."""

import operator
from typing import Annotated, Any, Literal, TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from src.agents.browser import BrowserAgent
from src.agents.computer import ComputerAgent
from src.agents.planner import PlannerAgent

logger = structlog.get_logger()


class AgentState(TypedDict):
    """State shared across all agents in the workflow."""

    # Input
    task: str  # Original task from user

    # Planning
    plan: dict[str, Any] | None  # Execution plan from planner
    current_step: int  # Current step index

    # Execution
    results: Annotated[list[dict[str, Any]], operator.add]  # Accumulated results
    context: dict[str, Any]  # Context for sharing data between steps

    # Status
    error: str | None  # Error message if any
    completed: bool  # Whether task is completed


# Agent instances (initialized on first use)
_planner: PlannerAgent | None = None
_computer_agent: ComputerAgent | None = None
_browser_agent: BrowserAgent | None = None


def _get_planner() -> PlannerAgent:
    """Get or create planner agent."""
    global _planner
    if _planner is None:
        _planner = PlannerAgent()
    return _planner


def _get_computer_agent() -> ComputerAgent:
    """Get or create computer agent."""
    global _computer_agent
    if _computer_agent is None:
        _computer_agent = ComputerAgent()
    return _computer_agent


def _get_browser_agent() -> BrowserAgent:
    """Get or create browser agent."""
    global _browser_agent
    if _browser_agent is None:
        _browser_agent = BrowserAgent()
    return _browser_agent


async def planner_node(state: AgentState) -> AgentState:
    """
    Planner node - creates execution plan.

    Args:
        state: Current agent state

    Returns:
        Updated state with plan
    """
    logger.info("planner_node_executing", task=state["task"])

    try:
        planner = _get_planner()
        plan = await planner.create_plan(state["task"], state.get("context"))

        logger.info("plan_created", steps=len(plan.get("plan", [])))

        return {
            **state,
            "plan": plan,
            "current_step": 0,
            "error": None,
        }

    except Exception as e:
        logger.error("planner_node_failed", error=str(e))
        return {
            **state,
            "error": f"Planning failed: {str(e)}",
            "completed": True,
        }


async def computer_node(state: AgentState) -> AgentState:
    """
    Computer control node - executes computer automation tasks.

    Args:
        state: Current agent state

    Returns:
        Updated state with execution result
    """
    logger.info("computer_node_executing", step=state["current_step"])

    try:
        computer_agent = _get_computer_agent()

        # Get current step
        plan = state.get("plan", {})
        steps = plan.get("plan", [])

        if state["current_step"] >= len(steps):
            logger.warning("no_more_steps")
            return {**state, "completed": True}

        step = steps[state["current_step"]]

        # Execute step
        result = await computer_agent.execute_step(step)

        logger.info("computer_step_completed", step=state["current_step"], success=result["success"])

        # Update context with result
        context = state.get("context", {})
        context[f"step_{state['current_step']}"] = result

        return {
            **state,
            "results": [result],
            "context": context,
            "current_step": state["current_step"] + 1,
            "error": None if result["success"] else result.get("error"),
        }

    except Exception as e:
        logger.error("computer_node_failed", error=str(e))
        return {
            **state,
            "results": [{"output": str(e), "success": False, "agent": "computer"}],
            "error": f"Computer automation failed: {str(e)}",
        }


async def browser_node(state: AgentState) -> AgentState:
    """
    Browser automation node - executes web tasks.

    Args:
        state: Current agent state

    Returns:
        Updated state with execution result
    """
    logger.info("browser_node_executing", step=state["current_step"])

    try:
        browser_agent = _get_browser_agent()

        # Get current step
        plan = state.get("plan", {})
        steps = plan.get("plan", [])

        if state["current_step"] >= len(steps):
            logger.warning("no_more_steps")
            return {**state, "completed": True}

        step = steps[state["current_step"]]

        # Execute step
        result = await browser_agent.execute_step(step)

        logger.info("browser_step_completed", step=state["current_step"], success=result["success"])

        # Update context with result
        context = state.get("context", {})
        context[f"step_{state['current_step']}"] = result

        return {
            **state,
            "results": [result],
            "context": context,
            "current_step": state["current_step"] + 1,
            "error": None if result["success"] else result.get("error"),
        }

    except Exception as e:
        logger.error("browser_node_failed", error=str(e))
        return {
            **state,
            "results": [{"output": str(e), "success": False, "agent": "browser"}],
            "error": f"Browser automation failed: {str(e)}",
        }


async def synthesizer_node(state: AgentState) -> AgentState:
    """
    Synthesizer node - combines results and generates final output.

    Args:
        state: Current agent state

    Returns:
        Updated state with synthesis
    """
    logger.info("synthesizer_node_executing", result_count=len(state["results"]))

    try:
        planner = _get_planner()
        synthesis = await planner.synthesize_results(state["task"], state["results"])

        logger.info("synthesis_completed")

        return {
            **state,
            "results": [synthesis],
            "completed": True,
            "error": None,
        }

    except Exception as e:
        logger.error("synthesizer_node_failed", error=str(e))
        return {
            **state,
            "error": f"Synthesis failed: {str(e)}",
            "completed": True,
        }


def route_from_planner(state: AgentState) -> Literal["computer", "browser", "synthesizer", END]:
    """
    Route from planner to appropriate agent or end.

    Args:
        state: Current agent state

    Returns:
        Next node to execute
    """
    # Check for errors
    if state.get("error"):
        logger.warning("routing_to_end_due_to_error", error=state["error"])
        return END

    # Check if plan exists
    plan = state.get("plan")
    if not plan or "plan" not in plan:
        logger.warning("routing_to_end_no_plan")
        return END

    steps = plan["plan"]
    current_step = state.get("current_step", 0)

    # Check if all steps completed
    if current_step >= len(steps):
        logger.info("routing_to_synthesizer_all_steps_complete")
        return "synthesizer"

    # Route to appropriate agent
    step = steps[current_step]
    agent = step.get("agent", "computer").lower()

    if agent == "browser":
        logger.info("routing_to_browser", step=current_step)
        return "browser"
    elif agent == "computer":
        logger.info("routing_to_computer", step=current_step)
        return "computer"
    elif agent == "planner":
        logger.info("routing_to_synthesizer", step=current_step)
        return "synthesizer"
    else:
        # Default to computer
        logger.warning("routing_to_computer_default", step=current_step, agent=agent)
        return "computer"


def route_from_agent(state: AgentState) -> Literal["computer", "browser", "synthesizer", END]:
    """
    Route from agent execution back to planner or next agent.

    Args:
        state: Current agent state

    Returns:
        Next node to execute
    """
    # Check for errors
    if state.get("error"):
        logger.warning("routing_to_synthesizer_due_to_error", error=state["error"])
        return "synthesizer"

    # Check if completed
    if state.get("completed"):
        logger.info("routing_to_end_completed")
        return END

    # Continue with next step
    plan = state.get("plan")
    if not plan or "plan" not in plan:
        logger.warning("routing_to_synthesizer_no_plan")
        return "synthesizer"

    steps = plan["plan"]
    current_step = state.get("current_step", 0)

    # Check if all steps completed
    if current_step >= len(steps):
        logger.info("routing_to_synthesizer_steps_complete")
        return "synthesizer"

    # Route to next agent
    step = steps[current_step]
    agent = step.get("agent", "computer").lower()

    if agent == "browser":
        logger.info("routing_to_browser_next_step", step=current_step)
        return "browser"
    elif agent == "computer":
        logger.info("routing_to_computer_next_step", step=current_step)
        return "computer"
    else:
        logger.info("routing_to_synthesizer_default", step=current_step)
        return "synthesizer"


def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow.

    Returns:
        Compiled StateGraph workflow
    """
    logger.info("creating_workflow")

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("computer", computer_node)
    workflow.add_node("browser", browser_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Set entry point
    workflow.add_edge(START, "planner")

    # Add conditional edges from planner
    workflow.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "computer": "computer",
            "browser": "browser",
            "synthesizer": "synthesizer",
            END: END,
        },
    )

    # Add conditional edges from agents back to routing
    workflow.add_conditional_edges(
        "computer",
        route_from_agent,
        {
            "computer": "computer",
            "browser": "browser",
            "synthesizer": "synthesizer",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "browser",
        route_from_agent,
        {
            "computer": "computer",
            "browser": "browser",
            "synthesizer": "synthesizer",
            END: END,
        },
    )

    # Synthesizer always goes to end
    workflow.add_edge("synthesizer", END)

    logger.info("workflow_created")

    return workflow.compile()
