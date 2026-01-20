# Autonomous Multi-LLM Orchestration System

A production-grade AI orchestration system using **LangGraph** that coordinates multiple LLMs for computer control and web automation tasks.

## 🚀 Features

- **Multi-LLM Orchestration**: Coordinates Claude (planning), Grok (computer control), and DeepSeek (browser automation)
- **LangGraph Workflow**: Modern StateGraph-based orchestration (not legacy LangChain)
- **Safety-First**: Comprehensive safety controls for computer automation
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Async-Native**: Built with async/await for optimal performance
- **Production-Ready**: Proper error handling, logging, and type hints

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Task Input                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             Planner Agent (Claude 3.5)                  │
│  • Task decomposition                                   │
│  • Agent delegation                                     │
│  • Result synthesis                                     │
└───────────┬─────────────────────────────────────────────┘
            │
            ├─────────────┬─────────────┐
            ▼             ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌─────────────────┐
│   Computer    │ │    Browser    │ │  Synthesizer   │
│  Agent (Grok) │ │ Agent (Deep-  │ │ (Claude 3.5)   │
│               │ │  Seek Local)  │ │                │
│ • pyautogui   │ │ • Playwright  │ │ • Results      │
│ • OS control  │ │ • Web tasks   │ │   combination  │
│ • Screenshots │ │ • Scraping    │ │ • Final output │
└───────────────┘ └───────────────┘ └─────────────────┘
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11 or higher
- **RAM**: 16GB recommended (8GB minimum for local DeepSeek)
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)

### Required Software
1. **Python 3.11+**: [Download Python](https://www.python.org/downloads/)
2. **Ollama**: [Download Ollama](https://ollama.ai)
3. **Git**: For cloning the repository

### API Keys
- **Anthropic API Key**: [Get API key](https://console.anthropic.com/)
- **xAI API Key**: [Get API key](https://x.ai/)
- **DeepSeek API Key** (optional): Only if using API mode instead of local

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd agentic-system
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install in editable mode
pip install -e .

# For development (includes testing tools)
pip install -e ".[dev]"
```

### 4. Install Playwright Browsers

```bash
playwright install chromium
```

### 5. Pull Local DeepSeek Model

```bash
# Pull the DeepSeek model via Ollama
ollama pull deepseek-r1:8b

# Verify installation
ollama list
```

### 6. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Windows: notepad .env
# macOS/Linux: nano .env
```

Add your API keys to `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-xxxxx
XAI_API_KEY=xai-xxxxx
DEEPSEEK_MODE=local
SAFETY_MODE=strict
```

## 🎯 Usage

### Basic Usage

```bash
# Run with a task
python -m src.main "Research AI meetups in Salt Lake City"

# Interactive mode
python -m src.main
> Enter your task: Open notepad and write a summary
```

### Advanced Usage

```bash
# Run with permissive safety mode (use with caution!)
SAFETY_MODE=permissive python -m src.main "Your task here"

# Use DeepSeek API instead of local
DEEPSEEK_MODE=api python -m src.main "Your task here"

# Enable debug logging
LOG_LEVEL=DEBUG python -m src.main "Your task here"
```

### Example Tasks

#### Example 1: Web Research
```bash
python -m src.main "Research the top 3 AI news stories today and summarize them"
```

**Expected Flow:**
1. Planner creates research plan
2. Browser agent navigates to news sites
3. Browser agent extracts article information
4. Planner synthesizes findings into summary

#### Example 2: Computer Automation
```bash
python -m src.main "Open notepad, type 'Hello World', and save the file"
```

**Expected Flow:**
1. Planner breaks down into steps
2. Computer agent opens notepad
3. Computer agent types text
4. Computer agent saves file
5. Planner confirms completion

#### Example 3: Mixed Task
```bash
python -m src.main "Search for Python tutorials, summarize the top result, and save summary to a text file"
```

**Expected Flow:**
1. Browser agent searches and extracts content
2. Planner synthesizes summary
3. Computer agent creates and saves file

## 🔒 Safety Features

### Strict Mode (Default)
- **Blocked Actions**: `rm`, `del`, `format`, `shutdown`, dangerous commands
- **App Whitelist**: Only allowed apps (notepad, chrome, firefox, etc.)
- **Confirmation Required**: Prompts for file modifications
- **Screenshot Logging**: All actions captured for audit

### Permissive Mode
```bash
SAFETY_MODE=permissive python -m src.main "Your task"
```
- Relaxed app restrictions
- Still blocks dangerous system commands
- Use only when you trust the task completely

### Safety Validation

The system automatically:
- ✅ Validates all commands before execution
- ✅ Clamps screen coordinates to boundaries
- ✅ Logs all actions with timestamps
- ✅ Times out operations after 30 seconds
- ✅ Takes before/after screenshots

## 📁 Project Structure

```
agentic-system/
├── src/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings with validation
│   ├── llms.py            # LLM client initialization
│   ├── graph.py           # LangGraph workflow definition
│   ├── main.py            # Entry point
│   ├── tools/
│   │   ├── computer.py    # pyautogui tools with safety
│   │   └── browser.py     # Playwright tools
│   └── agents/
│       ├── planner.py     # Claude-based planner
│       ├── computer.py    # Grok-based computer agent
│       └── browser.py     # DeepSeek-based browser agent
├── tests/
│   └── test_tools.py      # Basic tool tests
├── logs/                  # Auto-created for screenshots
├── pyproject.toml         # Modern Python packaging
├── .env.example           # Environment template
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_tools.py -v
```

## 🔧 Configuration Options

All configuration via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *required* | Claude API key |
| `XAI_API_KEY` | *required* | xAI/Grok API key |
| `DEEPSEEK_API_KEY` | optional | DeepSeek API key (if using API mode) |
| `DEEPSEEK_MODE` | `local` | `local` or `api` |
| `SAFETY_MODE` | `strict` | `strict` or `permissive` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ACTION_TIMEOUT` | `30` | Timeout in seconds (1-300) |
| `SCREENSHOT_LOGGING` | `true` | Enable/disable screenshots |
| `REQUIRE_CONFIRMATION` | `true` | Prompt for confirmations |

## 🐛 Troubleshooting

### Ollama Not Running

**Error**: `Failed to initialize local DeepSeek via Ollama`

**Solution**:
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
# macOS/Linux: automatically starts
# Windows: run Ollama from Start menu

# Pull model again
ollama pull deepseek-r1:8b
```

### API Rate Limits

**Error**: `429 Too Many Requests`

**Solution**:
- Wait a few minutes and retry
- System has exponential backoff built-in
- Check your API quota/limits

### Browser Not Launching

**Error**: `Failed to initialize browser`

**Solution**:
```bash
# Reinstall Playwright browsers
playwright install chromium

# Or install all browsers
playwright install
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in the project root
cd agentic-system

# Install in editable mode
pip install -e .
```

### Screen Coordinates Out of Bounds

**Error**: `coordinates_clamped` warnings

**Solution**:
- This is normal - coordinates are auto-clamped
- Adjust `MAX_SCREEN_WIDTH`/`MAX_SCREEN_HEIGHT` in `.env` if needed

## 🚨 Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `SafetyError: Blocked dangerous action` | Tried to run blocked command | Use `SAFETY_MODE=permissive` or modify whitelist |
| `BrowserToolError: Element not found` | Selector doesn't exist | Try different selector or wait longer |
| `ComputerToolError: Click operation failed` | Invalid coordinates | Check screen size and coordinates |
| `LLMInitializationError` | Invalid API key | Verify API keys in `.env` |

## 📊 Logging

Logs are stored in `logs/` directory:
- `logs/screenshots/` - Action screenshots
- `logs/browser_screenshots/` - Browser screenshots

View logs in real-time:
```bash
# Enable debug logging
LOG_LEVEL=DEBUG python -m src.main "Your task"
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Format code (`black . && ruff check .`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- **LangGraph**: For the orchestration framework
- **Anthropic**: For Claude API
- **xAI**: For Grok API
- **DeepSeek**: For local reasoning model
- **Playwright**: For reliable browser automation
- **pyautogui**: For cross-platform computer control

## 📧 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check existing issues for solutions
- Review troubleshooting section above

---

**⚠️ Important Safety Notice**

This system can control your computer and interact with websites. Always:
- Run in `SAFETY_MODE=strict` for untrusted tasks
- Review tasks before executing
- Monitor execution in real-time
- Keep screenshots enabled for audit trails
- Use in a sandboxed environment when testing

**Never** run untrusted tasks with permissive mode enabled.
