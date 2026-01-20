"""Computer control agent using Grok for OS automation."""

import structlog
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from src.llms import get_grok_llm
from src.tools.computer import COMPUTER_TOOLS

logger = structlog.get_logger()

COMPUTER_AGENT_PROMPT = """You are a computer control specialist AI. Your role is to:

1. **Execute precise computer control tasks** using the available tools
2. **Operate safely** within security constraints
3. **Provide clear feedback** on actions taken

## Available Tools

You have access to these computer control tools:
- safe_click: Click at screen coordinates
- safe_type_text: Type text (with safety validation)
- safe_press_key: Press keyboard keys
- safe_move_mouse: Move mouse cursor
- safe_open_app: Open applications (whitelist in strict mode)
- take_screenshot: Capture current screen

## Safety Constraints

- You are running in {safety_mode} mode
- Dangerous operations (del, rm, format, etc.) are BLOCKED
- In strict mode, only whitelisted apps can be opened
- All actions are logged with screenshots
- Action timeout: {action_timeout} seconds

## Guidelines

1. **Be Precise**: Use exact coordinates and clear instructions
2. **Verify Actions**: Take screenshots to confirm results
3. **Handle Errors**: Provide clear error messages and alternatives
4. **Stay Safe**: Never attempt to bypass security controls
5. **Be Efficient**: Complete tasks with minimal steps

## Task Execution

When given a task:
1. Break it into clear steps
2. Execute each step methodically
3. Verify results when possible
4. Report success or failure clearly

Always prioritize safety and user intent.
"""


class ComputerAgent:
    """Computer control agent using Grok."""

    def __init__(self) -> None:
        """Initialize computer agent."""
        from src.config import get_settings

        self.settings = get_settings()
        self.llm = get_grok_llm()

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    COMPUTER_AGENT_PROMPT.format(
                        safety_mode=self.settings.safety_mode.value,
                        action_timeout=self.settings.action_timeout,
                    ),
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Create agent with tools
        self.agent = create_tool_calling_agent(self.llm, COMPUTER_TOOLS, self.prompt)

        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=COMPUTER_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
        )

        logger.info(
            "computer_agent_initialized",
            safety_mode=self.settings.safety_mode.value,
            tools=len(COMPUTER_TOOLS),
        )

    async def execute(self, task: str) -> dict[str, str]:
        """
        Execute a computer control task.

        Args:
            task: Task description

        Returns:
            dict: Execution result with output and status
        """
        try:
            logger.info("executing_computer_task", task=task)

            # Execute task
            result = await self.executor.ainvoke({"input": task})

            output = result.get("output", "")
            success = True

            logger.info("computer_task_completed", success=success)

            return {"output": output, "success": success, "agent": "computer"}

        except Exception as e:
            logger.error("computer_task_failed", error=str(e), task=task)
            return {
                "output": f"Computer control task failed: {str(e)}",
                "success": False,
                "agent": "computer",
                "error": str(e),
            }

    async def execute_step(self, step: dict) -> dict[str, str]:
        """
        Execute a single step from a plan.

        Args:
            step: Step dictionary with description and action

        Returns:
            dict: Execution result
        """
        description = step.get("description", "")
        logger.info("executing_computer_step", step=step.get("step"), description=description)

        return await self.execute(description)
