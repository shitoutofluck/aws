"""Planner agent using Claude for task decomposition and orchestration."""

import json
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.llms import get_claude_llm

logger = structlog.get_logger()

PLANNER_SYSTEM_PROMPT = """You are an expert AI task planner and orchestrator. Your role is to:

1. **Analyze tasks** and break them down into clear, executable steps
2. **Delegate work** to specialized agents:
   - **Computer Agent**: For OS-level automation (clicking, typing, opening apps, screenshots)
   - **Browser Agent**: For web tasks (navigation, scraping, form filling, data extraction)
3. **Synthesize results** from agent executions into coherent outputs
4. **Adapt plans** based on execution results and errors

## Delegation Guidelines

### Use Computer Agent for:
- Opening desktop applications
- Taking screenshots
- Mouse clicks and keyboard input at OS level
- File operations via GUI interaction
- Cross-application workflows

### Use Browser Agent for:
- Web navigation and research
- Data extraction from websites
- Form filling and submissions
- Link following and page analysis
- Web-based automation

## Output Format

When planning, respond with a JSON object:
```json
{
    "plan": [
        {
            "step": 1,
            "action": "browser_search | computer_control | synthesize",
            "description": "Clear description of what to do",
            "agent": "browser | computer | planner"
        }
    ],
    "reasoning": "Why this plan will accomplish the task"
}
```

When synthesizing results, provide a clear summary of what was accomplished.

## Important Guidelines

- Be specific in step descriptions
- Consider failure modes and provide alternatives
- Don't over-complicate - simpler plans are better
- Leverage each agent's strengths appropriately
- Always validate that results meet the original task requirements
"""


class PlannerAgent:
    """Planner agent for task decomposition and orchestration."""

    def __init__(self) -> None:
        """Initialize planner agent."""
        self.llm = get_claude_llm()
        logger.info("planner_agent_initialized")

    async def create_plan(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Create an execution plan for a task.

        Args:
            task: Task description from user
            context: Optional context from previous steps

        Returns:
            dict: Plan with steps and reasoning
        """
        try:
            logger.info("creating_plan", task=task)

            # Build context message if available
            context_str = ""
            if context:
                context_str = f"\n\nContext from previous steps:\n{json.dumps(context, indent=2)}"

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", PLANNER_SYSTEM_PROMPT),
                    (
                        "human",
                        "Create a detailed execution plan for this task:\n\n{task}{context}\n\n"
                        "Respond with a JSON object containing the plan and reasoning.",
                    ),
                ]
            )

            chain = prompt | self.llm
            response = await chain.ainvoke({"task": task, "context": context_str})

            # Extract JSON from response
            content = response.content if hasattr(response, "content") else str(response)

            # Try to parse JSON from response
            try:
                # Look for JSON in the response
                import re

                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    plan = json.loads(json_match.group())
                else:
                    # Fallback: create a simple plan
                    plan = {
                        "plan": [
                            {
                                "step": 1,
                                "action": "execute",
                                "description": task,
                                "agent": "computer",
                            }
                        ],
                        "reasoning": "Direct execution of task",
                    }
            except json.JSONDecodeError:
                logger.warning("failed_to_parse_plan_json", content=content[:200])
                # Create fallback plan
                plan = {
                    "plan": [
                        {"step": 1, "action": "execute", "description": task, "agent": "computer"}
                    ],
                    "reasoning": "Fallback plan due to parsing error",
                }

            logger.info("plan_created", steps=len(plan.get("plan", [])))
            return plan

        except Exception as e:
            logger.error("plan_creation_failed", error=str(e), task=task)
            raise

    async def synthesize_results(
        self, task: str, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Synthesize results from agent executions.

        Args:
            task: Original task
            results: List of execution results from agents

        Returns:
            dict: Synthesized output with summary and conclusions
        """
        try:
            logger.info("synthesizing_results", task=task, result_count=len(results))

            results_str = json.dumps(results, indent=2)

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", PLANNER_SYSTEM_PROMPT),
                    (
                        "human",
                        "Original task: {task}\n\n"
                        "Agent execution results:\n{results}\n\n"
                        "Provide a clear, concise summary of what was accomplished and the key findings. "
                        "If the task involved research or data gathering, present the information in a "
                        "well-organized format.",
                    ),
                ]
            )

            chain = prompt | self.llm
            response = await chain.ainvoke({"task": task, "results": results_str})

            content = response.content if hasattr(response, "content") else str(response)

            synthesis = {
                "summary": content,
                "task": task,
                "steps_completed": len(results),
                "success": True,
            }

            logger.info("synthesis_completed")
            return synthesis

        except Exception as e:
            logger.error("synthesis_failed", error=str(e), task=task)
            raise

    async def evaluate_result(
        self, task: str, plan: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluate if a step's result is satisfactory.

        Args:
            task: Original task
            plan: The execution plan
            result: Result from agent execution

        Returns:
            dict: Evaluation with success status and next action
        """
        try:
            logger.info("evaluating_result", task=task)

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", PLANNER_SYSTEM_PROMPT),
                    (
                        "human",
                        "Original task: {task}\n\n"
                        "Execution plan: {plan}\n\n"
                        "Result: {result}\n\n"
                        "Evaluate if this result is satisfactory and whether we should:\n"
                        "1. Continue to next step\n"
                        "2. Retry this step with modifications\n"
                        "3. Mark task as complete\n"
                        "4. Report failure\n\n"
                        "Respond with JSON: "
                        '{{"success": true/false, "action": "continue|retry|complete|fail", '
                        '"reasoning": "explanation"}}',
                    ),
                ]
            )

            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "task": task,
                    "plan": json.dumps(plan, indent=2),
                    "result": json.dumps(result, indent=2),
                }
            )

            content = response.content if hasattr(response, "content") else str(response)

            # Try to parse JSON
            try:
                import re

                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    evaluation = json.loads(json_match.group())
                else:
                    evaluation = {"success": True, "action": "continue", "reasoning": "Default"}
            except json.JSONDecodeError:
                logger.warning("failed_to_parse_evaluation_json")
                evaluation = {"success": True, "action": "continue", "reasoning": "Parsing failed"}

            logger.info("evaluation_completed", action=evaluation.get("action"))
            return evaluation

        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            raise
