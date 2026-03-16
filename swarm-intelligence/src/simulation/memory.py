"""Local persistent memory system for simulation agents.

Replaces MiroFish's Zep Cloud memory with a local JSON-backed store.
Provides episodic memory (events), semantic memory (learned facts),
and working memory (current context) for each agent.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class MemoryEntry:
    """A single memory entry."""

    timestamp: str
    round_num: int
    content: str
    memory_type: str  # "episodic", "semantic", "working"
    importance: float = 0.5  # 0-1: how important this memory is
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "round_num": self.round_num,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "metadata": self.metadata,
        }


class AgentMemory:
    """Memory store for a single agent."""

    def __init__(self, agent_key: str, max_entries: int = 100) -> None:
        self.agent_key = agent_key
        self.max_entries = max_entries
        self.episodic: list[MemoryEntry] = []  # What happened
        self.semantic: list[MemoryEntry] = []   # What was learned
        self.working: list[MemoryEntry] = []    # Current context

    def add_episodic(self, content: str, round_num: int, importance: float = 0.5, **metadata: Any) -> None:
        """Record an event that happened."""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            round_num=round_num,
            content=content,
            memory_type="episodic",
            importance=importance,
            metadata=metadata,
        )
        self.episodic.append(entry)
        self._prune(self.episodic)

    def add_semantic(self, content: str, round_num: int, importance: float = 0.7, **metadata: Any) -> None:
        """Record a learned fact or insight."""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            round_num=round_num,
            content=content,
            memory_type="semantic",
            importance=importance,
            metadata=metadata,
        )
        self.semantic.append(entry)
        self._prune(self.semantic)

    def set_working(self, content: str, round_num: int, **metadata: Any) -> None:
        """Update working memory (current context)."""
        entry = MemoryEntry(
            timestamp=datetime.now().isoformat(),
            round_num=round_num,
            content=content,
            memory_type="working",
            importance=1.0,
            metadata=metadata,
        )
        # Working memory is small — keep only recent entries
        self.working.append(entry)
        if len(self.working) > 10:
            self.working = self.working[-10:]

    def get_recent(self, n: int = 5) -> list[MemoryEntry]:
        """Get most recent memories across all types."""
        all_memories = self.episodic + self.semantic + self.working
        all_memories.sort(key=lambda m: m.timestamp, reverse=True)
        return all_memories[:n]

    def get_context_string(self, max_entries: int = 5) -> str:
        """Get a formatted context string for LLM prompts."""
        recent = self.get_recent(max_entries)
        if not recent:
            return "No memories yet."

        lines = []
        for m in recent:
            lines.append(f"[Round {m.round_num}] ({m.memory_type}) {m.content}")
        return "\n".join(lines)

    def _prune(self, memory_list: list[MemoryEntry]) -> None:
        """Remove least important entries when over capacity."""
        if len(memory_list) > self.max_entries:
            memory_list.sort(key=lambda m: m.importance, reverse=True)
            del memory_list[self.max_entries:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_key": self.agent_key,
            "episodic": [e.to_dict() for e in self.episodic],
            "semantic": [e.to_dict() for e in self.semantic],
            "working": [e.to_dict() for e in self.working],
        }


class MemoryManager:
    """Manages memory for all agents in a simulation.

    Provides persistent storage via JSON files so simulations
    can be paused and resumed.
    """

    def __init__(self, storage_dir: str = "data/memory") -> None:
        self.storage_dir = storage_dir
        self.agents: dict[str, AgentMemory] = {}
        os.makedirs(storage_dir, exist_ok=True)
        logger.info("memory_manager_initialized", storage_dir=storage_dir)

    def get_or_create(self, agent_key: str) -> AgentMemory:
        """Get or create memory for an agent."""
        if agent_key not in self.agents:
            self.agents[agent_key] = AgentMemory(agent_key)
        return self.agents[agent_key]

    def record_action(
        self,
        agent_key: str,
        round_num: int,
        action: str,
        content: str,
        importance: float = 0.5,
    ) -> None:
        """Record an agent's action as episodic memory."""
        memory = self.get_or_create(agent_key)
        memory.add_episodic(
            f"Action: {action} - {content}",
            round_num=round_num,
            importance=importance,
        )

    def record_event(
        self,
        round_num: int,
        event: str,
        importance: float = 0.8,
    ) -> None:
        """Record a global event in all agents' memories."""
        for agent_key, memory in self.agents.items():
            memory.add_semantic(
                f"Event: {event}",
                round_num=round_num,
                importance=importance,
            )

    def save(self, simulation_id: str) -> str:
        """Save all agent memories to disk.

        Returns:
            Path to saved file.
        """
        filepath = os.path.join(self.storage_dir, f"{simulation_id}_memory.json")
        data = {
            "simulation_id": simulation_id,
            "saved_at": datetime.now().isoformat(),
            "agents": {key: mem.to_dict() for key, mem in self.agents.items()},
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("memory_saved", path=filepath, agents=len(self.agents))
        return filepath

    def load(self, simulation_id: str) -> bool:
        """Load agent memories from disk.

        Returns:
            True if loaded successfully.
        """
        filepath = os.path.join(self.storage_dir, f"{simulation_id}_memory.json")
        if not os.path.exists(filepath):
            logger.warning("memory_file_not_found", path=filepath)
            return False

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for agent_key, agent_data in data.get("agents", {}).items():
            memory = self.get_or_create(agent_key)
            for entry_data in agent_data.get("episodic", []):
                memory.episodic.append(MemoryEntry(**entry_data))
            for entry_data in agent_data.get("semantic", []):
                memory.semantic.append(MemoryEntry(**entry_data))
            for entry_data in agent_data.get("working", []):
                memory.working.append(MemoryEntry(**entry_data))

        logger.info("memory_loaded", path=filepath, agents=len(self.agents))
        return True
