"""Browser automation tools using Playwright."""

import asyncio
from typing import Any

import structlog
from langchain_core.tools import tool
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class BrowserToolError(Exception):
    """Raised when a browser tool operation fails."""

    pass


class BrowserManager:
    """Manages Playwright browser instance lifecycle."""

    def __init__(self) -> None:
        """Initialize browser manager."""
        self.playwright: Any = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._lock = asyncio.Lock()

    async def initialize(self, headless: bool = False) -> None:
        """
        Initialize browser instance.

        Args:
            headless: Whether to run browser in headless mode.
        """
        async with self._lock:
            if self.browser is not None:
                logger.debug("browser_already_initialized")
                return

            try:
                logger.info("initializing_playwright_browser", headless=headless)
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(headless=headless)
                self.context = await self.browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                )
                self.page = await self.context.new_page()
                logger.info("browser_initialized_successfully")
            except Exception as e:
                logger.error("browser_initialization_failed", error=str(e))
                await self.cleanup()
                raise BrowserToolError(f"Failed to initialize browser: {e}") from e

    async def cleanup(self) -> None:
        """Clean up browser resources."""
        async with self._lock:
            try:
                if self.page:
                    await self.page.close()
                if self.context:
                    await self.context.close()
                if self.browser:
                    await self.browser.close()
                if self.playwright:
                    await self.playwright.stop()

                self.page = None
                self.context = None
                self.browser = None
                self.playwright = None

                logger.info("browser_cleanup_complete")
            except Exception as e:
                logger.warning("browser_cleanup_failed", error=str(e))

    async def get_page(self) -> Page:
        """
        Get current page instance, initializing if necessary.

        Returns:
            Page: Current Playwright page instance.
        """
        if self.page is None:
            await self.initialize()

        if self.page is None:
            raise BrowserToolError("Failed to get page instance")

        return self.page


# Global browser manager instance
_browser_manager = BrowserManager()


async def _ensure_browser() -> Page:
    """
    Ensure browser is initialized and return page.

    Returns:
        Page: Current Playwright page instance.
    """
    return await _browser_manager.get_page()


# Tool input schemas
class NavigateInput(BaseModel):
    """Input schema for navigation."""

    url: str = Field(..., min_length=1, description="URL to navigate to")
    wait_for: str = Field(
        default="load", description="Wait for state: 'load', 'domcontentloaded', or 'networkidle'"
    )


class ClickElementInput(BaseModel):
    """Input schema for clicking elements."""

    selector: str = Field(..., description="CSS selector or text to click")
    timeout: int = Field(default=30000, ge=1000, le=60000, description="Timeout in milliseconds")


class TypeTextInput(BaseModel):
    """Input schema for typing in elements."""

    selector: str = Field(..., description="CSS selector for input element")
    text: str = Field(..., description="Text to type")
    clear_first: bool = Field(default=True, description="Clear existing text first")


class ExtractTextInput(BaseModel):
    """Input schema for extracting text."""

    selector: str = Field(
        default="body", description="CSS selector for element to extract text from"
    )


class ExtractLinksInput(BaseModel):
    """Input schema for extracting links."""

    selector: str = Field(default="a", description="CSS selector for link elements")


class WaitForElementInput(BaseModel):
    """Input schema for waiting for elements."""

    selector: str = Field(..., description="CSS selector to wait for")
    timeout: int = Field(default=30000, ge=1000, le=60000, description="Timeout in milliseconds")
    state: str = Field(
        default="visible", description="Element state: 'visible', 'hidden', or 'attached'"
    )


class ScreenshotInput(BaseModel):
    """Input schema for screenshots."""

    full_page: bool = Field(default=True, description="Capture full page or just viewport")


# Tool definitions
@tool(args_schema=NavigateInput)
async def browser_navigate(url: str, wait_for: str = "load") -> str:
    """
    Navigate to a URL in the browser.

    This tool navigates to the specified URL and waits for the page to load.
    Supports different wait strategies for dynamic content.

    Args:
        url: URL to navigate to (must include protocol)
        wait_for: Wait strategy ('load', 'domcontentloaded', or 'networkidle')

    Returns:
        str: Success message with final URL
    """
    try:
        # Validate URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        page = await _ensure_browser()

        logger.info("navigating_to_url", url=url, wait_for=wait_for)
        response = await page.goto(url, wait_until=wait_for)

        if response is None:
            raise BrowserToolError(f"Failed to navigate to {url}")

        final_url = page.url
        title = await page.title()

        logger.info("navigation_successful", url=final_url, title=title, status=response.status)
        return f"Navigated to: {final_url}\nTitle: {title}\nStatus: {response.status}"
    except Exception as e:
        logger.error("navigation_failed", error=str(e), url=url)
        raise BrowserToolError(f"Navigation failed: {e}") from e


@tool(args_schema=ClickElementInput)
async def browser_click(selector: str, timeout: int = 30000) -> str:
    """
    Click an element in the browser.

    This tool finds and clicks an element using CSS selector or text content.
    Waits for the element to be visible and clickable.

    Args:
        selector: CSS selector or text to click
        timeout: Maximum wait time in milliseconds

    Returns:
        str: Success message
    """
    try:
        page = await _ensure_browser()

        logger.info("clicking_element", selector=selector, timeout=timeout)

        # Try multiple selector strategies
        try:
            await page.click(selector, timeout=timeout)
        except Exception:
            # Try text-based selection as fallback
            await page.click(f"text={selector}", timeout=timeout)

        logger.info("click_successful", selector=selector)
        return f"Successfully clicked: {selector}"
    except Exception as e:
        logger.error("click_failed", error=str(e), selector=selector)
        raise BrowserToolError(f"Click failed: {e}") from e


@tool(args_schema=TypeTextInput)
async def browser_type(selector: str, text: str, clear_first: bool = True) -> str:
    """
    Type text into an input element.

    This tool finds an input element and types text into it.
    Can optionally clear existing text first.

    Args:
        selector: CSS selector for input element
        text: Text to type
        clear_first: Whether to clear existing text first

    Returns:
        str: Success message
    """
    try:
        page = await _ensure_browser()

        logger.info("typing_in_element", selector=selector, length=len(text), clear_first=clear_first)

        if clear_first:
            await page.fill(selector, text)
        else:
            await page.type(selector, text)

        logger.info("type_successful", selector=selector)
        return f"Successfully typed {len(text)} characters into: {selector}"
    except Exception as e:
        logger.error("type_failed", error=str(e), selector=selector)
        raise BrowserToolError(f"Type failed: {e}") from e


@tool(args_schema=ExtractTextInput)
async def browser_extract_text(selector: str = "body") -> str:
    """
    Extract text content from the page.

    This tool extracts visible text from the specified element.
    Useful for scraping content and reading page information.

    Args:
        selector: CSS selector for element (default: entire page body)

    Returns:
        str: Extracted text content
    """
    try:
        page = await _ensure_browser()

        logger.info("extracting_text", selector=selector)
        element = await page.query_selector(selector)

        if element is None:
            raise BrowserToolError(f"Element not found: {selector}")

        text = await element.inner_text()
        logger.info("text_extracted", selector=selector, length=len(text))

        return text
    except Exception as e:
        logger.error("text_extraction_failed", error=str(e), selector=selector)
        raise BrowserToolError(f"Text extraction failed: {e}") from e


@tool(args_schema=ExtractLinksInput)
async def browser_extract_links(selector: str = "a") -> str:
    """
    Extract links from the page.

    This tool finds all links matching the selector and extracts their URLs and text.
    Useful for discovering navigation options and scraping links.

    Args:
        selector: CSS selector for link elements (default: all <a> tags)

    Returns:
        str: Formatted list of links with text and URLs
    """
    try:
        page = await _ensure_browser()

        logger.info("extracting_links", selector=selector)
        elements = await page.query_selector_all(selector)

        links = []
        for element in elements:
            href = await element.get_attribute("href")
            text = await element.inner_text()
            if href:
                # Make relative URLs absolute
                absolute_url = page.url
                if href.startswith("/"):
                    from urllib.parse import urljoin

                    absolute_url = urljoin(page.url, href)
                elif href.startswith("http"):
                    absolute_url = href
                else:
                    from urllib.parse import urljoin

                    absolute_url = urljoin(page.url, href)

                links.append({"text": text.strip(), "url": absolute_url})

        logger.info("links_extracted", count=len(links))

        # Format output
        if not links:
            return "No links found"

        result = f"Found {len(links)} link(s):\n\n"
        for i, link in enumerate(links[:50], 1):  # Limit to 50 links
            result += f"{i}. {link['text']}\n   URL: {link['url']}\n\n"

        if len(links) > 50:
            result += f"... and {len(links) - 50} more links"

        return result
    except Exception as e:
        logger.error("link_extraction_failed", error=str(e), selector=selector)
        raise BrowserToolError(f"Link extraction failed: {e}") from e


@tool(args_schema=WaitForElementInput)
async def browser_wait_for(selector: str, timeout: int = 30000, state: str = "visible") -> str:
    """
    Wait for an element to appear.

    This tool waits for an element to reach a specific state.
    Useful for handling dynamic content and AJAX loading.

    Args:
        selector: CSS selector to wait for
        timeout: Maximum wait time in milliseconds
        state: Element state ('visible', 'hidden', or 'attached')

    Returns:
        str: Success message
    """
    try:
        page = await _ensure_browser()

        logger.info("waiting_for_element", selector=selector, state=state, timeout=timeout)
        await page.wait_for_selector(selector, state=state, timeout=timeout)

        logger.info("element_found", selector=selector, state=state)
        return f"Element found: {selector} (state: {state})"
    except Exception as e:
        logger.error("wait_failed", error=str(e), selector=selector)
        raise BrowserToolError(f"Wait failed: {e}") from e


@tool(args_schema=ScreenshotInput)
async def browser_screenshot(full_page: bool = True) -> str:
    """
    Take a screenshot of the current page.

    This tool captures the current browser view and saves it.
    Useful for debugging and verification.

    Args:
        full_page: Whether to capture full page or just viewport

    Returns:
        str: Path to saved screenshot
    """
    try:
        from datetime import datetime
        from pathlib import Path

        page = await _ensure_browser()

        screenshot_dir = Path("logs/browser_screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = screenshot_dir / f"screenshot_{timestamp}.png"

        logger.info("taking_screenshot", full_page=full_page, path=str(filename))
        await page.screenshot(path=str(filename), full_page=full_page)

        logger.info("screenshot_saved", path=str(filename))
        return f"Screenshot saved to: {filename}"
    except Exception as e:
        logger.error("screenshot_failed", error=str(e))
        raise BrowserToolError(f"Screenshot failed: {e}") from e


@tool
async def browser_get_url() -> str:
    """
    Get the current page URL.

    Returns:
        str: Current page URL and title
    """
    try:
        page = await _ensure_browser()
        url = page.url
        title = await page.title()

        logger.info("got_current_url", url=url, title=title)
        return f"Current URL: {url}\nTitle: {title}"
    except Exception as e:
        logger.error("get_url_failed", error=str(e))
        raise BrowserToolError(f"Get URL failed: {e}") from e


@tool
async def browser_go_back() -> str:
    """
    Navigate back in browser history.

    Returns:
        str: Success message with new URL
    """
    try:
        page = await _ensure_browser()

        logger.info("navigating_back")
        await page.go_back()

        url = page.url
        logger.info("navigation_back_successful", url=url)
        return f"Navigated back to: {url}"
    except Exception as e:
        logger.error("go_back_failed", error=str(e))
        raise BrowserToolError(f"Go back failed: {e}") from e


# Export all tools
BROWSER_TOOLS = [
    browser_navigate,
    browser_click,
    browser_type,
    browser_extract_text,
    browser_extract_links,
    browser_wait_for,
    browser_screenshot,
    browser_get_url,
    browser_go_back,
]


# Cleanup function
async def cleanup_browser() -> None:
    """Clean up browser resources."""
    await _browser_manager.cleanup()
