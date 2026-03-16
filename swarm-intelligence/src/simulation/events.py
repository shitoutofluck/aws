"""Variable injection system (God's Eye View).

Allows real-time injection of events into running simulations,
observing how the digital world reorganizes in response.
Inspired by MiroFish's dynamic parameter injection.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class SimulationEvent:
    """An event to be injected into the simulation."""

    event_id: str
    description: str
    impact_type: str = "general"  # "economic", "political", "social", "general"
    magnitude: float = 0.5       # 0-1: how strongly it affects opinions
    target_groups: list[int] | None = None  # Specific groups affected, or None for all
    round_injected: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "description": self.description,
            "impact_type": self.impact_type,
            "magnitude": self.magnitude,
            "target_groups": self.target_groups,
            "round_injected": self.round_injected,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class EventQueue:
    """Manages event injection for simulations.

    Supports:
    - Immediate injection (next round)
    - Scheduled injection (specific round)
    - Event templates for common scenarios
    """

    def __init__(self) -> None:
        self.pending: list[SimulationEvent] = []
        self.history: list[SimulationEvent] = []
        logger.info("event_queue_initialized")

    def inject(
        self,
        description: str,
        impact_type: str = "general",
        magnitude: float = 0.5,
        target_groups: list[int] | None = None,
    ) -> SimulationEvent:
        """Inject an event for the next round.

        Args:
            description: Natural language event description.
            impact_type: Category of impact.
            magnitude: Strength of impact (0-1).
            target_groups: Specific groups to affect, or None for all.

        Returns:
            The created SimulationEvent.
        """
        import uuid

        event = SimulationEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            description=description,
            impact_type=impact_type,
            magnitude=magnitude,
            target_groups=target_groups,
        )
        self.pending.append(event)
        logger.info("event_queued", event_id=event.event_id, description=description)
        return event

    def get_pending(self, round_num: int) -> list[SimulationEvent]:
        """Get and consume all pending events for a round.

        Args:
            round_num: Current round number.

        Returns:
            List of events to process this round.
        """
        events = list(self.pending)
        for evt in events:
            evt.round_injected = round_num
        self.history.extend(events)
        self.pending.clear()
        return events

    def save(self, filepath: str) -> None:
        """Save event history to file."""
        data = {
            "pending": [e.to_dict() for e in self.pending],
            "history": [e.to_dict() for e in self.history],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def create_economic_event(description: str, magnitude: float = 0.7) -> dict[str, Any]:
        """Template for economic events."""
        return {
            "description": description,
            "impact_type": "economic",
            "magnitude": magnitude,
        }

    @staticmethod
    def create_political_event(description: str, magnitude: float = 0.6) -> dict[str, Any]:
        """Template for political events."""
        return {
            "description": description,
            "impact_type": "political",
            "magnitude": magnitude,
        }
