"""Agent profile data structures with rich personality modeling.

Each agent in the simulation has a unique biography, personality traits,
social connections, and behavioral parameters — inspired by MiroFish's
OASIS Agent Profile system.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# MBTI personality types for diverse agent behaviors
MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP",
]

# Behavioral archetypes that influence simulation dynamics
ARCHETYPES = [
    "Leader",        # Influences others, forms groups
    "Follower",      # Adopts majority opinion
    "Contrarian",    # Opposes popular opinion
    "Analyst",       # Data-driven, slow to change
    "Activist",      # Strong opinions, vocal
    "Moderate",      # Balanced, seeks compromise
    "Innovator",     # Early adopter, creative
    "Traditionalist", # Resists change
    "Connector",     # Links groups, high social ties
    "Observer",      # Low activity, gathers info
]

# Opinion scale: -1.0 (strongly against) to +1.0 (strongly for)
DEFAULT_OPINION_RANGE = (-0.5, 0.5)


@dataclass
class AgentProfile:
    """Rich agent profile for simulation participants."""

    # Identity
    agent_id: int
    name: str
    entity_type: str = "Person"

    # Biography (LLM-generated from knowledge graph)
    bio: str = ""
    persona: str = ""  # Detailed persona description

    # Personality
    mbti: str = ""
    archetype: str = ""
    openness: float = 0.5       # 0-1: willingness to change opinion
    assertiveness: float = 0.5  # 0-1: tendency to voice opinions
    sociability: float = 0.5    # 0-1: tendency to interact with others
    influence: float = 0.5      # 0-1: ability to change others' opinions

    # Demographics
    age: int | None = None
    gender: str | None = None
    profession: str | None = None
    country: str | None = None

    # Social network
    connections: list[int] = field(default_factory=list)  # Agent IDs
    group_id: int | None = None  # Cluster/group membership

    # Simulation state
    opinion: float = 0.0        # Current opinion (-1 to 1)
    activity_level: float = 0.5  # 0-1: how active per round
    memory_key: str = ""        # Key into memory system

    # Source
    source_entity_uuid: str | None = None
    source_entity_type: str | None = None
    interested_topics: list[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "bio": self.bio,
            "persona": self.persona,
            "mbti": self.mbti,
            "archetype": self.archetype,
            "openness": self.openness,
            "assertiveness": self.assertiveness,
            "sociability": self.sociability,
            "influence": self.influence,
            "age": self.age,
            "gender": self.gender,
            "profession": self.profession,
            "country": self.country,
            "connections": self.connections,
            "group_id": self.group_id,
            "opinion": self.opinion,
            "activity_level": self.activity_level,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def random_personality() -> dict[str, Any]:
        """Generate random personality traits for diversity."""
        archetype = random.choice(ARCHETYPES)

        # Personality traits correlated with archetype
        trait_presets: dict[str, dict[str, float]] = {
            "Leader":        {"openness": 0.7, "assertiveness": 0.9, "sociability": 0.8, "influence": 0.9},
            "Follower":      {"openness": 0.6, "assertiveness": 0.2, "sociability": 0.6, "influence": 0.2},
            "Contrarian":    {"openness": 0.3, "assertiveness": 0.8, "sociability": 0.4, "influence": 0.5},
            "Analyst":       {"openness": 0.5, "assertiveness": 0.4, "sociability": 0.3, "influence": 0.6},
            "Activist":      {"openness": 0.4, "assertiveness": 0.9, "sociability": 0.9, "influence": 0.7},
            "Moderate":      {"openness": 0.7, "assertiveness": 0.5, "sociability": 0.6, "influence": 0.4},
            "Innovator":     {"openness": 0.9, "assertiveness": 0.6, "sociability": 0.5, "influence": 0.6},
            "Traditionalist": {"openness": 0.2, "assertiveness": 0.6, "sociability": 0.5, "influence": 0.5},
            "Connector":     {"openness": 0.8, "assertiveness": 0.5, "sociability": 0.95, "influence": 0.6},
            "Observer":      {"openness": 0.6, "assertiveness": 0.1, "sociability": 0.2, "influence": 0.2},
        }

        base = trait_presets.get(archetype, {})

        # Add noise for diversity
        def noisy(val: float) -> float:
            return max(0.0, min(1.0, val + random.gauss(0, 0.1)))

        return {
            "mbti": random.choice(MBTI_TYPES),
            "archetype": archetype,
            "openness": noisy(base.get("openness", 0.5)),
            "assertiveness": noisy(base.get("assertiveness", 0.5)),
            "sociability": noisy(base.get("sociability", 0.5)),
            "influence": noisy(base.get("influence", 0.5)),
            "opinion": random.uniform(*DEFAULT_OPINION_RANGE),
            "activity_level": max(0.1, min(1.0, random.gauss(0.5, 0.2))),
        }
