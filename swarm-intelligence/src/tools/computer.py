"""Computer control tools with safety wrappers using pyautogui."""

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import pyautogui
import structlog
from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel, Field

from src.config import SafetyMode, get_settings

logger = structlog.get_logger()

# Safety configurations
BLOCKED_ACTIONS = [
    "del ",
    "rm ",
    "format",
    "shutdown",
    "reboot",
    "registry",
    "sudo",
    "admin",
    "password",
    "passwd",
    "chmod 777",
    "chown",
    "dd if=",
    "mkfs",
    "> /dev/",
    "kill -9",
    "killall",
    "systemctl",
    "service",
]

ALLOWED_APPS = [
    "notepad",
    "notepad.exe",
    "chrome",
    "firefox",
    "safari",
    "code",
    "vscode",
    "terminal",
    "iterm",
    "cmd",
    "powershell",
    "explorer",
    "finder",
    "calculator",
    "calc",
]


class SafetyError(Exception):
    """Raised when a dangerous action is attempted."""

    pass


class ComputerToolError(Exception):
    """Raised when a computer tool operation fails."""

    pass


def _validate_action(action: str, context: str = "") -> bool:
    """
    Validate action against blocked patterns.

    Args:
        action: The action string to validate.
        context: Additional context for logging.

    Returns:
        bool: True if action is safe.

    Raises:
        SafetyError: If action contains blocked patterns.
    """
    action_lower = action.lower()

    for blocked in BLOCKED_ACTIONS:
        if blocked in action_lower:
            logger.warning(
                "blocked_dangerous_action",
                action=action,
                blocked_pattern=blocked,
                context=context,
            )
            raise SafetyError(
                f"Blocked dangerous action containing '{blocked}'. "
                f"Action: {action[:100]}... "
                f"This action is prohibited for security reasons."
            )

    logger.debug("action_validated", action=action[:100], context=context)
    return True


def _validate_app(app_name: str) -> bool:
    """
    Validate application name against allowed list.

    Args:
        app_name: Application name to validate.

    Returns:
        bool: True if application is allowed.

    Raises:
        SafetyError: If application is not in allowed list.
    """
    settings = get_settings()

    # In permissive mode, allow all apps
    if settings.safety_mode == SafetyMode.PERMISSIVE:
        logger.debug("app_validation_skipped_permissive_mode", app=app_name)
        return True

    # In strict mode, check against whitelist
    app_lower = app_name.lower()
    if not any(allowed in app_lower for allowed in ALLOWED_APPS):
        logger.warning("blocked_disallowed_app", app=app_name)
        raise SafetyError(
            f"Application '{app_name}' is not in the allowed list: {ALLOWED_APPS}. "
            f"To allow this app, run with SAFETY_MODE=permissive (use with caution)."
        )

    logger.debug("app_validated", app=app_name)
    return True


def _save_screenshot(prefix: str = "action") -> Path:
    """
    Save a screenshot for logging purposes.

    Args:
        prefix: Filename prefix.

    Returns:
        Path: Path to saved screenshot.
    """
    settings = get_settings()

    if not settings.screenshot_logging:
        logger.debug("screenshot_logging_disabled")
        return Path("/dev/null")

    try:
        screenshot_dir = Path("logs/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = screenshot_dir / f"{prefix}_{timestamp}.png"

        screenshot = pyautogui.screenshot()
        screenshot.save(filename)

        logger.debug("screenshot_saved", path=str(filename))
        return filename
    except Exception as e:
        logger.warning("screenshot_save_failed", error=str(e))
        return Path("/dev/null")


def _clamp_coordinates(x: int, y: int) -> tuple[int, int]:
    """
    Clamp coordinates to screen bounds.

    Args:
        x: X coordinate.
        y: Y coordinate.

    Returns:
        tuple[int, int]: Clamped coordinates.
    """
    settings = get_settings()
    screen_width, screen_height = pyautogui.size()

    # Use the smaller of actual screen size or configured max
    max_width = min(screen_width, settings.max_screen_width)
    max_height = min(screen_height, settings.max_screen_height)

    clamped_x = max(0, min(x, max_width - 1))
    clamped_y = max(0, min(y, max_height - 1))

    if (clamped_x, clamped_y) != (x, y):
        logger.warning(
            "coordinates_clamped",
            original=(x, y),
            clamped=(clamped_x, clamped_y),
            screen_size=(screen_width, screen_height),
        )

    return clamped_x, clamped_y


# Tool input schemas
class ClickInput(BaseModel):
    """Input schema for click action."""

    x: int = Field(..., ge=0, description="Screen X coordinate")
    y: int = Field(..., ge=0, description="Screen Y coordinate")
    clicks: int = Field(default=1, ge=1, le=3, description="Number of clicks (1-3)")
    button: Literal["left", "right", "middle"] = Field(
        default="left", description="Mouse button to click"
    )


class TypeTextInput(BaseModel):
    """Input schema for typing text."""

    text: str = Field(..., min_length=1, max_length=1000, description="Text to type")
    interval: float = Field(
        default=0.05, ge=0.0, le=1.0, description="Interval between keystrokes in seconds"
    )


class PressKeyInput(BaseModel):
    """Input schema for pressing keys."""

    key: str = Field(..., description="Key to press (e.g., 'enter', 'tab', 'ctrl')")
    presses: int = Field(default=1, ge=1, le=10, description="Number of times to press")


class MoveMouseInput(BaseModel):
    """Input schema for moving mouse."""

    x: int = Field(..., ge=0, description="Target X coordinate")
    y: int = Field(..., ge=0, description="Target Y coordinate")
    duration: float = Field(default=0.5, ge=0.0, le=2.0, description="Movement duration in seconds")


class OpenAppInput(BaseModel):
    """Input schema for opening applications."""

    app_name: str = Field(..., min_length=1, max_length=100, description="Application name to open")


# Tool definitions
@tool(args_schema=ClickInput)
def safe_click(x: int, y: int, clicks: int = 1, button: str = "left") -> str:
    """
    Click at screen coordinates with safety bounds checking.

    This tool moves the mouse to the specified coordinates and performs a click.
    Coordinates are automatically clamped to screen boundaries.
    Screenshots are taken before the action if enabled.

    Args:
        x: X coordinate on screen
        y: Y coordinate on screen
        clicks: Number of clicks (1-3)
        button: Mouse button ('left', 'right', or 'middle')

    Returns:
        str: Success message with action details
    """
    settings = get_settings()

    try:
        # Save screenshot before action
        _save_screenshot("before_click")

        # Clamp coordinates
        x, y = _clamp_coordinates(x, y)

        # Perform click
        logger.info("performing_click", x=x, y=y, clicks=clicks, button=button)
        pyautogui.click(x=x, y=y, clicks=clicks, button=button)

        # Save screenshot after action
        _save_screenshot("after_click")

        return f"Successfully clicked at ({x}, {y}) with {button} button, {clicks} time(s)"
    except Exception as e:
        logger.error("click_failed", error=str(e), x=x, y=y)
        raise ComputerToolError(f"Click operation failed: {e}") from e


@tool(args_schema=TypeTextInput)
def safe_type_text(text: str, interval: float = 0.05) -> str:
    """
    Type text with safety validation.

    This tool types the specified text character by character.
    Dangerous commands and patterns are blocked in strict mode.

    Args:
        text: Text to type (max 1000 characters)
        interval: Delay between keystrokes in seconds

    Returns:
        str: Success message with character count
    """
    settings = get_settings()

    try:
        # Validate action in strict mode
        if settings.safety_mode == SafetyMode.STRICT:
            _validate_action(text, context="type_text")

        # Save screenshot before action
        _save_screenshot("before_type")

        # Type text
        logger.info("typing_text", length=len(text), interval=interval)
        pyautogui.write(text, interval=interval)

        # Save screenshot after action
        _save_screenshot("after_type")

        return f"Successfully typed {len(text)} characters"
    except SafetyError:
        raise
    except Exception as e:
        logger.error("type_text_failed", error=str(e))
        raise ComputerToolError(f"Type text operation failed: {e}") from e


@tool(args_schema=PressKeyInput)
def safe_press_key(key: str, presses: int = 1) -> str:
    """
    Press keyboard keys safely.

    This tool presses specified keyboard keys (e.g., 'enter', 'tab', 'ctrl+c').
    Supports modifier keys and key combinations.

    Args:
        key: Key to press (e.g., 'enter', 'tab', 'ctrl+c')
        presses: Number of times to press (1-10)

    Returns:
        str: Success message
    """
    try:
        # Save screenshot before action
        _save_screenshot("before_keypress")

        # Press key
        logger.info("pressing_key", key=key, presses=presses)
        for _ in range(presses):
            pyautogui.press(key)
            time.sleep(0.1)

        # Save screenshot after action
        _save_screenshot("after_keypress")

        return f"Successfully pressed '{key}' {presses} time(s)"
    except Exception as e:
        logger.error("press_key_failed", error=str(e), key=key)
        raise ComputerToolError(f"Press key operation failed: {e}") from e


@tool(args_schema=MoveMouseInput)
def safe_move_mouse(x: int, y: int, duration: float = 0.5) -> str:
    """
    Move mouse to coordinates smoothly.

    This tool moves the mouse cursor to the specified position.
    Coordinates are automatically clamped to screen boundaries.

    Args:
        x: Target X coordinate
        y: Target Y coordinate
        duration: Movement duration in seconds (0-2)

    Returns:
        str: Success message with coordinates
    """
    try:
        # Clamp coordinates
        x, y = _clamp_coordinates(x, y)

        # Move mouse
        logger.info("moving_mouse", x=x, y=y, duration=duration)
        pyautogui.moveTo(x, y, duration=duration)

        return f"Successfully moved mouse to ({x}, {y})"
    except Exception as e:
        logger.error("move_mouse_failed", error=str(e), x=x, y=y)
        raise ComputerToolError(f"Move mouse operation failed: {e}") from e


@tool(args_schema=OpenAppInput)
def safe_open_app(app_name: str) -> str:
    """
    Open an application safely.

    This tool opens the specified application.
    In strict mode, only whitelisted applications are allowed.

    Args:
        app_name: Name of the application to open

    Returns:
        str: Success message
    """
    settings = get_settings()

    try:
        # Validate app in strict mode
        if settings.safety_mode == SafetyMode.STRICT:
            _validate_app(app_name)

        # Save screenshot before action
        _save_screenshot("before_open_app")

        # Open application (cross-platform approach)
        logger.info("opening_app", app=app_name)

        # Use hotkey to open run dialog/spotlight and type app name
        import platform

        system = platform.system()

        if system == "Windows":
            pyautogui.hotkey("win", "r")
            time.sleep(0.5)
            pyautogui.write(app_name)
            pyautogui.press("enter")
        elif system == "Darwin":  # macOS
            pyautogui.hotkey("command", "space")
            time.sleep(0.5)
            pyautogui.write(app_name)
            time.sleep(0.3)
            pyautogui.press("enter")
        else:  # Linux
            pyautogui.hotkey("alt", "f2")
            time.sleep(0.5)
            pyautogui.write(app_name)
            pyautogui.press("enter")

        time.sleep(1)  # Wait for app to open

        # Save screenshot after action
        _save_screenshot("after_open_app")

        return f"Successfully opened application: {app_name}"
    except SafetyError:
        raise
    except Exception as e:
        logger.error("open_app_failed", error=str(e), app=app_name)
        raise ComputerToolError(f"Open app operation failed: {e}") from e


@tool
def take_screenshot() -> str:
    """
    Take a screenshot of the current screen.

    This tool captures the current screen state and saves it to the logs directory.
    Useful for debugging and verification.

    Returns:
        str: Path to saved screenshot
    """
    try:
        logger.info("taking_screenshot")
        screenshot_path = _save_screenshot("manual")
        return f"Screenshot saved to: {screenshot_path}"
    except Exception as e:
        logger.error("screenshot_failed", error=str(e))
        raise ComputerToolError(f"Screenshot operation failed: {e}") from e


# Export all tools
COMPUTER_TOOLS = [
    safe_click,
    safe_type_text,
    safe_press_key,
    safe_move_mouse,
    safe_open_app,
    take_screenshot,
]
