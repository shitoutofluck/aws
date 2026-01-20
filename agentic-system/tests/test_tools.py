"""Tests for computer and browser tools."""

import pytest
from unittest.mock import Mock, patch

from src.tools.computer import (
    _clamp_coordinates,
    _validate_action,
    _validate_app,
    SafetyError,
)


class TestComputerSafety:
    """Test computer control safety features."""

    def test_validate_action_blocks_dangerous_commands(self):
        """Test that dangerous commands are blocked."""
        dangerous_commands = [
            "rm -rf /",
            "del important.txt",
            "format c:",
            "shutdown now",
            "sudo rm -rf",
            "chmod 777 /etc/passwd",
        ]

        for cmd in dangerous_commands:
            with pytest.raises(SafetyError) as exc_info:
                _validate_action(cmd)
            assert "Blocked dangerous action" in str(exc_info.value)

    def test_validate_action_allows_safe_commands(self):
        """Test that safe commands pass validation."""
        safe_commands = [
            "open notepad",
            "type hello world",
            "click at coordinates",
            "take screenshot",
        ]

        for cmd in safe_commands:
            assert _validate_action(cmd) is True

    @patch("src.tools.computer.get_settings")
    def test_validate_app_strict_mode(self, mock_settings):
        """Test app validation in strict mode."""
        from src.config import SafetyMode

        mock_settings.return_value.safety_mode = SafetyMode.STRICT

        # Allowed apps should pass
        allowed_apps = ["notepad", "chrome", "firefox", "code"]
        for app in allowed_apps:
            assert _validate_app(app) is True

        # Disallowed apps should fail
        with pytest.raises(SafetyError) as exc_info:
            _validate_app("dangerous_app")
        assert "not in the allowed list" in str(exc_info.value)

    @patch("src.tools.computer.get_settings")
    def test_validate_app_permissive_mode(self, mock_settings):
        """Test app validation in permissive mode."""
        from src.config import SafetyMode

        mock_settings.return_value.safety_mode = SafetyMode.PERMISSIVE

        # All apps should pass in permissive mode
        assert _validate_app("any_app") is True
        assert _validate_app("dangerous_app") is True

    @patch("src.tools.computer.get_settings")
    @patch("src.tools.computer.pyautogui.size")
    def test_clamp_coordinates_within_bounds(self, mock_size, mock_settings):
        """Test coordinate clamping within screen bounds."""
        mock_size.return_value = (1920, 1080)
        mock_settings.return_value.max_screen_width = 3840
        mock_settings.return_value.max_screen_height = 2160

        # Coordinates within bounds should remain unchanged
        x, y = _clamp_coordinates(500, 500)
        assert (x, y) == (500, 500)

    @patch("src.tools.computer.get_settings")
    @patch("src.tools.computer.pyautogui.size")
    def test_clamp_coordinates_out_of_bounds(self, mock_size, mock_settings):
        """Test coordinate clamping for out-of-bounds values."""
        mock_size.return_value = (1920, 1080)
        mock_settings.return_value.max_screen_width = 3840
        mock_settings.return_value.max_screen_height = 2160

        # Coordinates beyond screen should be clamped
        x, y = _clamp_coordinates(5000, 5000)
        assert x <= 1920
        assert y <= 1080

        # Negative coordinates should be clamped to 0
        x, y = _clamp_coordinates(-100, -100)
        assert (x, y) == (0, 0)


class TestConfig:
    """Test configuration and settings."""

    def test_safety_mode_validation(self):
        """Test that safety mode is properly validated."""
        from src.config import SafetyMode

        assert SafetyMode.STRICT.value == "strict"
        assert SafetyMode.PERMISSIVE.value == "permissive"

    def test_deepseek_mode_validation(self):
        """Test that DeepSeek mode is properly validated."""
        from src.config import DeepSeekMode

        assert DeepSeekMode.LOCAL.value == "local"
        assert DeepSeekMode.API.value == "api"

    @patch.dict("os.environ", {
        "ANTHROPIC_API_KEY": "test-key",
        "XAI_API_KEY": "test-key",
        "SAFETY_MODE": "strict",
        "LOG_LEVEL": "INFO",
    }, clear=True)
    def test_settings_defaults(self):
        """Test that settings have correct defaults."""
        from src.config import Settings, reset_settings

        # Reset settings to ensure clean state
        reset_settings()

        settings = Settings()

        assert settings.safety_mode.value == "strict"
        assert settings.deepseek_mode.value == "local"
        assert settings.log_level == "INFO"
        assert settings.action_timeout == 30
        assert settings.screenshot_logging is True

    @patch.dict("os.environ", {
        "ANTHROPIC_API_KEY": "test-key",
        "XAI_API_KEY": "test-key",
        "LOG_LEVEL": "INVALID",
    }, clear=True)
    def test_settings_validation_invalid_log_level(self):
        """Test that invalid log levels are rejected."""
        from src.config import Settings, reset_settings
        from pydantic import ValidationError

        reset_settings()

        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "Log level must be one of" in str(exc_info.value)

    @patch.dict("os.environ", {
        "ANTHROPIC_API_KEY": "test-key",
        "XAI_API_KEY": "test-key",
        "ACTION_TIMEOUT": "500",
    }, clear=True)
    def test_settings_validation_invalid_timeout(self):
        """Test that invalid timeouts are rejected."""
        from src.config import Settings, reset_settings
        from pydantic import ValidationError

        reset_settings()

        with pytest.raises(ValidationError) as exc_info:
            Settings()
        assert "Action timeout must be between 1 and 300" in str(exc_info.value)


class TestBrowserTools:
    """Test browser automation tools."""

    @pytest.mark.asyncio
    async def test_browser_manager_initialization(self):
        """Test that browser manager can be initialized."""
        from src.tools.browser import BrowserManager

        manager = BrowserManager()
        assert manager.browser is None
        assert manager.page is None

    @pytest.mark.asyncio
    async def test_browser_cleanup(self):
        """Test browser cleanup."""
        from src.tools.browser import cleanup_browser

        # Should not raise even if browser not initialized
        await cleanup_browser()


@pytest.fixture(autouse=True)
def reset_test_environment():
    """Reset environment before each test."""
    from src.config import reset_settings

    reset_settings()
    yield
    reset_settings()
