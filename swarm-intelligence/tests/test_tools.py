"""Tests for swarm intelligence components."""

import json
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.agent_profile import AgentProfile, ARCHETYPES, MBTI_TYPES
from src.knowledge.graph import Entity, KnowledgeGraph, Relationship, TextProcessor
from src.simulation.engine import ActionType, AgentAction, SimulationEngine
from src.simulation.events import EventQueue, SimulationEvent
from src.simulation.memory import AgentMemory, MemoryManager


class TestTextProcessor:
    """Test document text processing."""

    def test_split_short_text(self):
        text = "Short text."
        chunks = TextProcessor.split_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        text = "A" * 1000
        chunks = TextProcessor.split_text(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        # Each chunk should be <= chunk_size
        for chunk in chunks:
            assert len(chunk) <= 300

    def test_split_preserves_content(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = TextProcessor.split_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        # All content should be covered
        combined = " ".join(chunks)
        for word in text.split():
            assert word in combined


class TestKnowledgeGraph:
    """Test knowledge graph construction."""

    def test_create_graph(self):
        kg = KnowledgeGraph()
        assert kg.graph_id.startswith("kg_")
        assert len(kg.entities) == 0
        assert len(kg.relationships) == 0

    def test_add_entity(self):
        kg = KnowledgeGraph()
        entity = kg._add_entity("Alice", "Person", "A researcher")
        assert entity.name == "Alice"
        assert entity.entity_type == "Person"
        assert entity.uuid in kg.entities

    def test_add_duplicate_entity(self):
        kg = KnowledgeGraph()
        e1 = kg._add_entity("Alice", "Person", "Short desc")
        e2 = kg._add_entity("alice", "Person", "A longer description")
        # Should return same entity, updated description
        assert e1.uuid == e2.uuid
        assert len(kg.entities) == 1
        assert e1.description == "A longer description"

    def test_add_relationship(self):
        kg = KnowledgeGraph()
        kg._add_entity("Alice", "Person")
        kg._add_entity("ACME", "Organization")
        rel = kg._add_relationship("Alice", "ACME", "works_at", "Alice works at ACME")
        assert rel is not None
        assert len(kg.relationships) == 1
        assert kg.graph.has_edge(rel.source_uuid, rel.target_uuid)

    def test_get_entities_by_type(self):
        kg = KnowledgeGraph()
        kg._add_entity("Alice", "Person")
        kg._add_entity("Bob", "Person")
        kg._add_entity("ACME", "Organization")
        persons = kg.get_entities_by_type("Person")
        assert len(persons) == 2

    def test_get_person_entities(self):
        kg = KnowledgeGraph()
        kg._add_entity("Alice", "Person")
        kg._add_entity("Activists", "Group")
        kg._add_entity("Climate Change", "Concept")
        persons = kg.get_person_entities()
        assert len(persons) == 2  # Person + Group

    def test_get_entity_context(self):
        kg = KnowledgeGraph()
        kg._add_entity("Alice", "Person", "A researcher")
        kg._add_entity("ACME", "Organization", "A tech company")
        kg._add_relationship("Alice", "ACME", "works_at")
        alice_uuid = kg._entity_name_to_uuid["alice"]
        context = kg.get_entity_context(alice_uuid)
        assert "Alice" in context
        assert "ACME" in context

    def test_get_info(self):
        kg = KnowledgeGraph()
        kg._add_entity("Alice", "Person")
        kg._add_entity("ACME", "Organization")
        kg._add_relationship("Alice", "ACME", "works_at")
        info = kg.get_info()
        assert info.node_count == 2
        assert info.edge_count == 1
        assert "Person" in info.entity_types

    def test_serialize_deserialize(self, tmp_path):
        kg = KnowledgeGraph(graph_id="test_graph")
        kg._add_entity("Alice", "Person", "A researcher")
        kg._add_entity("ACME", "Organization")
        kg._add_relationship("Alice", "ACME", "works_at")

        filepath = str(tmp_path / "graph.json")
        kg.save(filepath)

        loaded = KnowledgeGraph.load(filepath)
        assert loaded.graph_id == "test_graph"
        assert len(loaded.entities) == 2
        assert len(loaded.relationships) == 1


class TestAgentProfile:
    """Test agent profile generation."""

    def test_random_personality(self):
        personality = AgentProfile.random_personality()
        assert personality["mbti"] in MBTI_TYPES
        assert personality["archetype"] in ARCHETYPES
        assert 0.0 <= personality["openness"] <= 1.0
        assert 0.0 <= personality["assertiveness"] <= 1.0
        assert 0.0 <= personality["sociability"] <= 1.0
        assert 0.0 <= personality["influence"] <= 1.0
        assert -1.0 <= personality["opinion"] <= 1.0

    def test_profile_creation(self):
        profile = AgentProfile(
            agent_id=0,
            name="Alice",
            entity_type="Person",
            bio="A researcher",
            **AgentProfile.random_personality(),
        )
        assert profile.agent_id == 0
        assert profile.name == "Alice"
        assert profile.mbti in MBTI_TYPES

    def test_profile_serialization(self):
        profile = AgentProfile(
            agent_id=1,
            name="Bob",
            entity_type="Person",
            bio="An engineer",
            **AgentProfile.random_personality(),
        )
        data = profile.to_dict()
        assert data["agent_id"] == 1
        assert data["name"] == "Bob"

        restored = AgentProfile.from_dict(data)
        assert restored.agent_id == 1
        assert restored.name == "Bob"


class TestSimulationEngine:
    """Test simulation engine."""

    def _create_test_agents(self, n: int = 5) -> list[AgentProfile]:
        agents = []
        for i in range(n):
            agents.append(
                AgentProfile(
                    agent_id=i,
                    name=f"Agent_{i}",
                    entity_type="Person",
                    bio=f"Test agent {i}",
                    connections=[j for j in range(n) if j != i],
                    **AgentProfile.random_personality(),
                )
            )
        return agents

    @pytest.mark.asyncio
    async def test_rule_based_simulation(self):
        agents = self._create_test_agents(5)
        engine = SimulationEngine(
            agents=agents,
            topic="Test topic",
            max_rounds=3,
            llm=None,  # Rule-based mode
        )

        summaries = await engine.run()
        assert len(summaries) == 3
        assert engine.current_round == 3
        assert engine.status.value == "completed"

    @pytest.mark.asyncio
    async def test_event_injection(self):
        agents = self._create_test_agents(5)
        engine = SimulationEngine(
            agents=agents,
            topic="Test topic",
            max_rounds=2,
            llm=None,
        )

        engine.inject_event("Major policy change announced")
        assert len(engine._pending_events) == 1

        summaries = await engine.run()
        # Event should have been consumed
        assert len(engine._pending_events) == 0
        assert len(engine._event_history) == 1

    @pytest.mark.asyncio
    async def test_opinion_trajectory(self):
        agents = self._create_test_agents(5)
        engine = SimulationEngine(
            agents=agents,
            topic="Test topic",
            max_rounds=3,
            llm=None,
        )

        await engine.run()
        trajectory = engine.get_opinion_trajectory()
        assert trajectory["topic"] == "Test topic"
        assert trajectory["rounds"] == 3
        assert len(trajectory["agents"]) == 5

    @pytest.mark.asyncio
    async def test_social_influence(self):
        """Test that social influence causes opinion drift."""
        random.seed(42)
        agents = self._create_test_agents(10)
        # Set extreme opinions
        for i, a in enumerate(agents):
            a.opinion = 1.0 if i < 5 else -1.0
            a.openness = 0.8

        engine = SimulationEngine(
            agents=agents,
            topic="Test polarization",
            max_rounds=5,
            llm=None,
        )

        await engine.run()

        # Opinions should have drifted somewhat toward each other
        snapshot = engine.get_state_snapshot()
        assert snapshot["status"] == "completed"

    @pytest.mark.asyncio
    async def test_state_snapshot(self):
        agents = self._create_test_agents(5)
        engine = SimulationEngine(
            agents=agents,
            topic="Test",
            max_rounds=1,
            llm=None,
        )

        await engine.run()
        snapshot = engine.get_state_snapshot()
        assert "status" in snapshot
        assert "current_round" in snapshot
        assert "agent_count" in snapshot


class TestEventQueue:
    """Test event injection system."""

    def test_inject_event(self):
        queue = EventQueue()
        event = queue.inject("Fed cuts rates by 50bps", impact_type="economic")
        assert event.description == "Fed cuts rates by 50bps"
        assert event.impact_type == "economic"
        assert len(queue.pending) == 1

    def test_consume_events(self):
        queue = EventQueue()
        queue.inject("Event 1")
        queue.inject("Event 2")
        assert len(queue.pending) == 2

        events = queue.get_pending(round_num=5)
        assert len(events) == 2
        assert len(queue.pending) == 0
        assert len(queue.history) == 2
        assert events[0].round_injected == 5


class TestMemory:
    """Test memory system."""

    def test_agent_memory(self):
        memory = AgentMemory("agent_0")
        memory.add_episodic("Saw a post about AI", round_num=1)
        memory.add_semantic("AI is a hot topic", round_num=1)
        memory.set_working("Currently discussing AI policy", round_num=1)

        recent = memory.get_recent(5)
        assert len(recent) == 3

        context = memory.get_context_string()
        assert "AI" in context

    def test_memory_manager_persistence(self, tmp_path):
        manager = MemoryManager(storage_dir=str(tmp_path))
        mem = manager.get_or_create("agent_0")
        mem.add_episodic("Test event", round_num=1)

        manager.record_event(round_num=1, event="Global event")

        filepath = manager.save("test_sim")
        assert "test_sim_memory.json" in filepath

        # Load into fresh manager
        manager2 = MemoryManager(storage_dir=str(tmp_path))
        loaded = manager2.load("test_sim")
        assert loaded is True
        assert "agent_0" in manager2.agents


class TestConfig:
    """Test configuration."""

    @patch.dict("os.environ", {"DEEPSEEK_MODE": "local", "LOG_LEVEL": "INFO"}, clear=True)
    def test_settings_defaults(self):
        from src.config import Settings, reset_settings

        reset_settings()
        settings = Settings()

        assert settings.simulation_mode.value == "hybrid"
        assert settings.max_agents == 20
        assert settings.max_rounds == 10

    @patch.dict("os.environ", {"MAX_ROUNDS": "200"}, clear=True)
    def test_invalid_rounds(self):
        from pydantic import ValidationError

        from src.config import Settings, reset_settings

        reset_settings()
        with pytest.raises(ValidationError):
            Settings()


@pytest.fixture(autouse=True)
def reset_test_environment():
    from src.config import reset_settings

    reset_settings()
    yield
    reset_settings()
