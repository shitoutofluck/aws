# Swarm Intelligence Simulation Engine

A MiroFish-inspired swarm intelligence engine that constructs digital societies from documents, generates autonomous agents with unique personalities, and simulates social dynamics to produce predictions and analysis.

## Architecture

```
                        Document Input
                             |
                    [1. Knowledge Graph]
                     Local GraphRAG with
                     NetworkX + LLM entity
                     extraction
                             |
                    [2. Agent Generation]
                     Dynamic spawning from
                     graph nodes. MBTI,
                     archetypes, social ties
                             |
                    [3. Simulation Engine]
                     OASIS-style rounds:
                     opinions, influence,
                     group dynamics, events
                             |
                    [4. Report Generation]
                     LLM-powered analysis
                     with predictions and
                     opinion trajectories
                             |
                       Final Report
```

## Key Features

- **Local GraphRAG**: Extract entities and relationships from documents into a NetworkX knowledge graph (no cloud dependency)
- **Dynamic Agent Generation**: Spawn dozens of autonomous agents from graph nodes, each with unique biography, MBTI personality, archetype, and social connections
- **OASIS-style Simulation**: Round-based social dynamics with opinion formation, social influence, group behavior, and emergent phenomena
- **God's Eye View**: Inject real-time events during simulation (e.g., "Fed cuts rates by 50bps") and observe how the digital society reorganizes
- **Persistent Memory**: Local JSON-backed episodic, semantic, and working memory for each agent
- **Multi-LLM Support**: Claude (planning/reporting), Grok (analysis), DeepSeek via Ollama (fully offline)
- **Cost-Free Mode**: Runs entirely on local Ollama with rule-based fallbacks -- zero API costs
- **LangGraph Pipeline**: Modern StateGraph orchestration (not legacy LangChain)

## Quick Start

```bash
# Clone and setup
cd swarm-intelligence
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Pull local model (free, runs offline)
ollama pull deepseek-r1:8b

# Configure (optional -- works without API keys)
cp .env.example .env

# Run with a document
python -m src.main document.txt --topic "How will this policy affect public opinion?"

# Run with inline text
echo "The Federal Reserve announced..." | python -m src.main --topic "Market reaction"

# Inject events (God's Eye View)
python -m src.main doc.txt -t "Impact analysis" -e "Fed cuts rates 50bps" -e "Oil prices spike"

# Override simulation params
python -m src.main doc.txt -t "Topic" --rounds 20 --agents 50
```

## Simulation Modes

| Mode | Speed | Cost | Quality |
|------|-------|------|---------|
| `rule` | Fast | Free | Basic agent behaviors via archetype rules |
| `hybrid` | Medium | Low | LLM for key decisions, rules for the rest |
| `llm` | Slow | Higher | Full LLM-powered agent reasoning each round |

```bash
# Fast, free simulation
SIMULATION_MODE=rule python -m src.main doc.txt -t "Topic"

# Rich LLM simulation
SIMULATION_MODE=llm python -m src.main doc.txt -t "Topic"
```

## Project Structure

```
swarm-intelligence/
├── src/
│   ├── config.py                    # Pydantic settings
│   ├── llms.py                      # Multi-LLM initialization
│   ├── graph.py                     # LangGraph simulation pipeline
│   ├── main.py                      # CLI entry point
│   ├── knowledge/
│   │   ├── graph.py                 # Local GraphRAG (NetworkX + LLM)
│   │   └── ontology.py              # LLM-driven ontology generation
│   ├── agents/
│   │   ├── agent_profile.py         # Rich personality modeling
│   │   ├── agent_generator.py       # Dynamic agent spawning
│   │   └── report_agent.py          # Analysis and reporting
│   ├── simulation/
│   │   ├── engine.py                # OASIS-style simulation loop
│   │   ├── memory.py                # Persistent agent memory
│   │   └── events.py                # Variable injection (God's Eye)
│   └── tools/                       # (preserved from original)
│       ├── computer.py              # pyautogui safety tools
│       └── browser.py               # Playwright tools
├── tests/
│   └── test_tools.py                # Component tests
├── data/                            # Simulation output
├── pyproject.toml
├── .env.example
└── README.md
```

## How It Works

### 1. Knowledge Graph Construction
Documents are chunked and processed by an LLM to extract entities (Person, Organization, Event, Policy, etc.) and relationships. These form a NetworkX directed graph that serves as the "seed" for the digital society.

### 2. Agent Generation
Each entity in the graph becomes an autonomous agent with:
- **Biography**: LLM-generated from entity context and relationships
- **Personality**: MBTI type + behavioral archetype (Leader, Contrarian, Follower, etc.)
- **Traits**: Openness, assertiveness, sociability, influence (0-1 scales)
- **Social Network**: Connections derived from graph relationships + random small-world links
- **Group Membership**: Archetype-based clustering for emergent dynamics

### 3. Simulation Rounds
Each round:
1. Active agents (based on activity_level) decide actions
2. Actions include: post opinion, react to posts, discuss, do nothing
3. Social influence: opinions drift toward connected agents
4. Group dynamics: echo chambers, polarization, herd effects emerge
5. Events (if injected) shift opinions based on agent archetypes

### 4. Analysis & Reporting
A dedicated ReportAgent analyzes the simulation data:
- Opinion trajectories across rounds
- Group behavior patterns
- Event impact assessment
- Predictions for continued simulation
- Confidence levels

## Configuration

All via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMULATION_MODE` | `hybrid` | `llm`, `rule`, or `hybrid` |
| `MAX_AGENTS` | `20` | Max agents from graph |
| `MIN_AGENTS` | `5` | Min agents (synthetic fill) |
| `MAX_ROUNDS` | `10` | Simulation rounds |
| `USE_LLM_FOR_BIOS` | `true` | LLM agent biographies |
| `DEEPSEEK_MODE` | `local` | `local` (Ollama) or `api` |
| `ANTHROPIC_API_KEY` | *optional* | Claude for planning |
| `XAI_API_KEY` | *optional* | Grok for analysis |

## Testing

```bash
pip install -e ".[dev]"
pytest -v
```

## Example: Policy Impact Analysis

```bash
# Document: a news article about new regulation
python -m src.main regulation.txt \
  --topic "How will this regulation affect industry stakeholders?" \
  --event "Major company announces compliance plan" \
  --event "Opposition group files legal challenge" \
  --rounds 15 \
  --agents 30
```

**Output**: A structured report analyzing how different stakeholder archetypes (industry leaders, activists, moderates, etc.) react to the regulation and respond to unfolding events, with opinion trajectories and predictions.

## Comparison with MiroFish

| Feature | MiroFish | This System |
|---------|----------|-------------|
| Knowledge Graph | Zep Cloud (paid) | Local NetworkX (free) |
| Agent Memory | Zep Cloud | Local JSON files |
| Simulation | OASIS/CAMEL-AI | Custom OASIS-inspired engine |
| LLM Backend | OpenAI-compatible | Multi-LLM (Claude/Grok/Ollama) |
| Agent Profiles | Rich LLM-generated | Rich LLM-generated + archetypes |
| Variable Injection | Yes | Yes (God's Eye View) |
| Frontend | Vue dashboard | CLI (extensible) |
| Deployment | Docker | pip install (+ Docker optional) |
| Cost | API keys required | Fully free with Ollama |

## Potential Challenges

1. **LLM Costs**: Use `SIMULATION_MODE=rule` for free simulation, or `hybrid` for minimal API calls
2. **Consistency**: LLM agent responses may vary; rule-based fallback ensures deterministic behavior
3. **Hardware**: Local DeepSeek needs 8GB+ RAM; reduce `MAX_AGENTS` for lower-spec machines
4. **Scale**: For 100+ agents, use `rule` mode to avoid rate limits; `llm` mode best for <30 agents
