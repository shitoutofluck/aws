"""Browser automation agent using DeepSeek for web tasks."""

import structlog
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from src.llms import get_deepseek_llm
from src.tools.browser import BROWSER_TOOLS

logger = structlog.get_logger()

BROWSER_AGENT_PROMPT = """You are a web automation and research specialist AI. Your role is to:

1. **Navigate websites** efficiently and accurately
2. **Extract information** from web pages
3. **Interact with web elements** (forms, buttons, links)
4. **Research topics** by browsing multiple sources

## Available Tools

You have access to these browser automation tools:
- browser_navigate: Navigate to URLs
- browser_click: Click elements by selector or text
- browser_type: Type text into input fields
- browser_extract_text: Extract text content from pages
- browser_extract_links: Extract links from pages
- browser_wait_for: Wait for elements to appear
- browser_screenshot: Take screenshots
- browser_get_url: Get current URL and title
- browser_go_back: Navigate back in history

## Web Automation Best Practices

1. **Navigation**: Always navigate to a URL before interacting with elements
2. **Waiting**: Use browser_wait_for when dealing with dynamic content
3. **Selectors**: Prefer specific CSS selectors over text-based selection
4. **Extraction**: Extract relevant information systematically
5. **Verification**: Take screenshots to verify important steps

## Research Guidelines

When researching:
1. Navigate to authoritative sources
2. Extract key information systematically
3. Verify information across multiple sources when possible
4. Organize findings clearly
5. Provide source URLs for all information

## Error Handling

If an element is not found:
1. Wait for the element to load
2. Try alternative selectors
3. Check if you're on the correct page
4. Take a screenshot to diagnose

## Task Execution

When given a task:
1. Plan your navigation path
2. Execute steps methodically
3. Extract and organize information
4. Verify results
5. Provide clear, structured output

Focus on accuracy and completeness.
"""


class BrowserAgent:
    """Browser automation agent using DeepSeek."""

    def __init__(self) -> None:
        """Initialize browser agent."""
        self.llm = get_deepseek_llm()

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", BROWSER_AGENT_PROMPT),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Create agent with tools
        self.agent = create_tool_calling_agent(self.llm, BROWSER_TOOLS, self.prompt)

        # Create executor
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=BROWSER_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20,
        )

        logger.info("browser_agent_initialized", tools=len(BROWSER_TOOLS))

    async def execute(self, task: str) -> dict[str, str]:
        """
        Execute a browser automation task.

        Args:
            task: Task description

        Returns:
            dict: Execution result with output and status
        """
        try:
            logger.info("executing_browser_task", task=task)

            # Execute task
            result = await self.executor.ainvoke({"input": task})

            output = result.get("output", "")
            success = True

            logger.info("browser_task_completed", success=success)

            return {"output": output, "success": success, "agent": "browser"}

        except Exception as e:
            logger.error("browser_task_failed", error=str(e), task=task)
            return {
                "output": f"Browser automation task failed: {str(e)}",
                "success": False,
                "agent": "browser",
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
        logger.info("executing_browser_step", step=step.get("step"), description=description)

        return await self.execute(description)
