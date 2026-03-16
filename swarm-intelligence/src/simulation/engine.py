"""OASIS-style simulation engine for multi-agent social dynamics.

Runs round-based simulations where agents interact, form opinions,
create groups, and respond to events. Inspired by MiroFish's use of
the OASIS framework from CAMEL-AI.
"""

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.agents.agent_profile import AgentProfile

logger = structlog.get_logger()


class ActionType(str, Enum):
    """Actions agents can take each round."""

    POST_OPINION = "post_opinion"
    REACT_TO_POST = "react_to_post"
    DISCUSS = "discuss"
    FORM_GROUP = "form_group"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"
    DO_NOTHING = "do_nothing"
    CHANGE_OPINION = "change_opinion"


class SimulationStatus(str, Enum):
    """Simulation lifecycle states."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentAction:
    """Record of an agent's action in a round."""

    round_num: int
    timestamp: str
    agent_id: int
    agent_name: str
    action_type: ActionType
    content: str = ""
    target_agent_id: int | None = None
    opinion_before: float = 0.0
    opinion_after: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type.value,
            "content": self.content,
            "target_agent_id": self.target_agent_id,
            "opinion_before": round(self.opinion_before, 3),
            "opinion_after": round(self.opinion_after, 3),
            "metadata": self.metadata,
        }


@dataclass
class RoundSummary:
    """Summary of a simulation round."""

    round_num: int
    actions: list[AgentAction]
    avg_opinion: float
    opinion_std: float
    active_agents: int
    groups: dict[int, list[int]]  # group_id -> agent_ids
    events_injected: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_num": self.round_num,
            "action_count": len(self.actions),
            "avg_opinion": round(self.avg_opinion, 3),
            "opinion_std": round(self.opinion_std, 3),
            "active_agents": self.active_agents,
            "group_count": len(self.groups),
            "events_injected": self.events_injected,
        }


AGENT_ACTION_PROMPT = """You are {agent_name}, a participant in a social simulation about: {topic}

YOUR PROFILE:
- Bio: {bio}
- Persona: {persona}
- Personality: {archetype} ({mbti})
- Current opinion on the topic (scale -1 to +1): {opinion}
  (-1 = strongly against, 0 = neutral, +1 = strongly for)

RECENT EVENTS:
{recent_events}

POSTS FROM OTHER AGENTS THIS ROUND:
{other_posts}

YOUR CONNECTIONS' OPINIONS:
{connection_opinions}

Based on your personality and the information above, decide your action this round.
Respond with JSON only:
{{
    "action": "post_opinion | react_to_post | discuss | do_nothing",
    "content": "What you say or think (1-2 sentences)",
    "opinion_shift": <float between -0.3 and 0.3, how much your opinion changed>,
    "reasoning": "Brief internal reasoning for your action"
}}"""


class SimulationEngine:
    """Runs round-based social simulations with LLM-powered agents.

    Each round:
    1. Active agents decide actions based on personality + context
    2. Opinions update based on social influence
    3. Group dynamics evolve
    4. External events (if injected) are processed
    """

    def __init__(
        self,
        agents: list[AgentProfile],
        topic: str,
        max_rounds: int = 10,
        llm: Any = None,
    ) -> None:
        self.agents = {a.agent_id: a for a in agents}
        self.topic = topic
        self.max_rounds = max_rounds
        self.llm = llm
        self.current_round = 0
        self.status = SimulationStatus.CREATED

        # History
        self.round_summaries: list[RoundSummary] = []
        self.all_actions: list[AgentAction] = []
        self.opinion_history: list[dict[int, float]] = []

        # Event queue (God's Eye View)
        self._pending_events: list[str] = []
        self._event_history: list[tuple[int, str]] = []

        # Posts visible this round
        self._current_round_posts: list[dict[str, Any]] = []

        logger.info(
            "simulation_engine_created",
            agents=len(agents),
            topic=topic,
            max_rounds=max_rounds,
        )

    async def run(self) -> list[RoundSummary]:
        """Run the full simulation.

        Returns:
            List of RoundSummary for each round.
        """
        logger.info("simulation_starting", max_rounds=self.max_rounds)
        self.status = SimulationStatus.RUNNING

        try:
            for round_num in range(1, self.max_rounds + 1):
                if self.status != SimulationStatus.RUNNING:
                    break

                summary = await self._run_round(round_num)
                self.round_summaries.append(summary)

                logger.info(
                    "round_completed",
                    round=round_num,
                    avg_opinion=summary.avg_opinion,
                    active=summary.active_agents,
                    actions=len(summary.actions),
                )

            self.status = SimulationStatus.COMPLETED
            logger.info("simulation_completed", rounds=len(self.round_summaries))

        except Exception as e:
            self.status = SimulationStatus.FAILED
            logger.error("simulation_failed", error=str(e))
            raise

        return self.round_summaries

    async def _run_round(self, round_num: int) -> RoundSummary:
        """Execute a single simulation round."""
        self.current_round = round_num
        self._current_round_posts = []

        # Process any pending events
        round_events = list(self._pending_events)
        self._pending_events.clear()
        for event in round_events:
            self._event_history.append((round_num, event))

        # Determine which agents are active this round
        active_agents = [
            a for a in self.agents.values()
            if random.random() < a.activity_level
        ]

        # Shuffle to avoid order bias
        random.shuffle(active_agents)

        # Each active agent decides an action
        actions: list[AgentAction] = []
        for agent in active_agents:
            action = await self._agent_decide(agent, round_num, round_events)
            actions.append(action)
            self.all_actions.append(action)

            # If agent posted, add to visible posts
            if action.action_type in (ActionType.POST_OPINION, ActionType.DISCUSS):
                self._current_round_posts.append({
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "content": action.content,
                    "opinion": agent.opinion,
                })

        # Apply social influence (opinion contagion)
        self._apply_social_influence()

        # Record opinion snapshot
        snapshot = {a_id: a.opinion for a_id, a in self.agents.items()}
        self.opinion_history.append(snapshot)

        # Compute stats
        opinions = [a.opinion for a in self.agents.values()]
        avg_opinion = sum(opinions) / len(opinions) if opinions else 0.0
        opinion_std = (sum((o - avg_opinion) ** 2 for o in opinions) / len(opinions)) ** 0.5 if opinions else 0.0

        # Get current groups
        groups: dict[int, list[int]] = {}
        for a in self.agents.values():
            gid = a.group_id or 0
            groups.setdefault(gid, []).append(a.agent_id)

        return RoundSummary(
            round_num=round_num,
            actions=actions,
            avg_opinion=avg_opinion,
            opinion_std=opinion_std,
            active_agents=len(active_agents),
            groups=groups,
            events_injected=round_events,
        )

    async def _agent_decide(
        self,
        agent: AgentProfile,
        round_num: int,
        events: list[str],
    ) -> AgentAction:
        """Have an agent decide its action for this round."""
        opinion_before = agent.opinion

        # If we have an LLM, use it for rich decision-making
        if self.llm:
            try:
                action = await self._llm_decide(agent, round_num, events)
                return action
            except Exception as e:
                logger.warning("llm_decide_failed", agent=agent.name, error=str(e))

        # Fallback: rule-based decision
        return self._rule_based_decide(agent, round_num, events)

    async def _llm_decide(
        self,
        agent: AgentProfile,
        round_num: int,
        events: list[str],
    ) -> AgentAction:
        """Use LLM to make agent decisions (richer simulation)."""
        # Build context
        recent_events = "\n".join(events) if events else "No special events this round."

        other_posts = ""
        if self._current_round_posts:
            posts = self._current_round_posts[-10:]  # Last 10 posts
            other_posts = "\n".join(
                f"- {p['agent_name']}: \"{p['content']}\" (opinion: {p['opinion']:.2f})"
                for p in posts
            )
        else:
            other_posts = "No posts yet this round."

        # Connection opinions
        conn_opinions = ""
        if agent.connections:
            conn_lines = []
            for conn_id in agent.connections[:5]:
                conn = self.agents.get(conn_id)
                if conn:
                    conn_lines.append(f"- {conn.name}: opinion {conn.opinion:.2f}")
            conn_opinions = "\n".join(conn_lines) if conn_lines else "No close connections."
        else:
            conn_opinions = "No close connections."

        prompt = AGENT_ACTION_PROMPT.format(
            agent_name=agent.name,
            topic=self.topic,
            bio=agent.bio,
            persona=agent.persona,
            archetype=agent.archetype,
            mbti=agent.mbti,
            opinion=f"{agent.opinion:.2f}",
            recent_events=recent_events,
            other_posts=other_posts,
            connection_opinions=conn_opinions,
        )

        response = await self.llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Parse response
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {"action": "do_nothing", "content": "", "opinion_shift": 0}

        # Apply opinion shift
        shift = float(data.get("opinion_shift", 0))
        shift = max(-0.3, min(0.3, shift))  # Clamp
        agent.opinion = max(-1.0, min(1.0, agent.opinion + shift))

        action_map = {
            "post_opinion": ActionType.POST_OPINION,
            "react_to_post": ActionType.REACT_TO_POST,
            "discuss": ActionType.DISCUSS,
            "do_nothing": ActionType.DO_NOTHING,
        }

        return AgentAction(
            round_num=round_num,
            timestamp=datetime.now().isoformat(),
            agent_id=agent.agent_id,
            agent_name=agent.name,
            action_type=action_map.get(data.get("action", "do_nothing"), ActionType.DO_NOTHING),
            content=data.get("content", ""),
            opinion_before=opinion_before,
            opinion_after=agent.opinion,
            metadata={"reasoning": data.get("reasoning", "")},
        )

    def _rule_based_decide(
        self,
        agent: AgentProfile,
        round_num: int,
        events: list[str],
    ) -> AgentAction:
        """Fallback rule-based decision when LLM unavailable."""
        opinion_before = agent.opinion

        # Event response: shift opinion based on archetype
        if events:
            if agent.archetype in ("Leader", "Activist"):
                shift = random.uniform(-0.2, 0.2)
            elif agent.archetype in ("Follower", "Moderate"):
                # Move toward average opinion of connections
                if agent.connections:
                    conn_opinions = [
                        self.agents[c].opinion
                        for c in agent.connections
                        if c in self.agents
                    ]
                    if conn_opinions:
                        avg = sum(conn_opinions) / len(conn_opinions)
                        shift = (avg - agent.opinion) * agent.openness * 0.3
                    else:
                        shift = 0
                else:
                    shift = random.uniform(-0.05, 0.05)
            elif agent.archetype == "Contrarian":
                # Move against the crowd
                all_opinions = [a.opinion for a in self.agents.values()]
                avg = sum(all_opinions) / len(all_opinions)
                shift = -(avg - agent.opinion) * 0.2
            else:
                shift = random.uniform(-0.1, 0.1)

            agent.opinion = max(-1.0, min(1.0, agent.opinion + shift))

        # Decide action based on assertiveness
        if random.random() < agent.assertiveness:
            action_type = ActionType.POST_OPINION
            content = f"[{agent.archetype}] Opinion on {self.topic}: {'positive' if agent.opinion > 0 else 'negative' if agent.opinion < 0 else 'neutral'} ({agent.opinion:.2f})"
        elif random.random() < agent.sociability and self._current_round_posts:
            action_type = ActionType.REACT_TO_POST
            content = f"Reacting to discussion about {self.topic}"
        else:
            action_type = ActionType.DO_NOTHING
            content = ""

        return AgentAction(
            round_num=round_num,
            timestamp=datetime.now().isoformat(),
            agent_id=agent.agent_id,
            agent_name=agent.name,
            action_type=action_type,
            content=content,
            opinion_before=opinion_before,
            opinion_after=agent.opinion,
        )

    def _apply_social_influence(self) -> None:
        """Apply social influence: agents' opinions drift toward their connections.

        This creates emergent phenomena like:
        - Echo chambers within groups
        - Opinion polarization between groups
        - Herd effects when influential agents shift
        """
        opinion_updates: dict[int, float] = {}

        for agent in self.agents.values():
            if not agent.connections:
                continue

            # Weighted average of connection opinions
            influences = []
            for conn_id in agent.connections:
                conn = self.agents.get(conn_id)
                if conn:
                    # Weight by connection's influence and agent's openness
                    weight = conn.influence * agent.openness
                    influences.append((conn.opinion, weight))

            if not influences:
                continue

            total_weight = sum(w for _, w in influences)
            if total_weight > 0:
                weighted_avg = sum(o * w for o, w in influences) / total_weight
                # Small drift toward connection average
                drift = (weighted_avg - agent.opinion) * 0.05
                opinion_updates[agent.agent_id] = max(
                    -1.0, min(1.0, agent.opinion + drift)
                )

        # Apply all updates atomically
        for agent_id, new_opinion in opinion_updates.items():
            self.agents[agent_id].opinion = new_opinion

    def inject_event(self, event: str) -> None:
        """Inject a variable/event into the simulation (God's Eye View).

        Args:
            event: Natural language description of the event.
                   e.g., "Fed cuts rates by 50bps"
        """
        self._pending_events.append(event)
        logger.info("event_injected", event=event, pending=len(self._pending_events))

    def pause(self) -> None:
        """Pause the simulation."""
        self.status = SimulationStatus.PAUSED
        logger.info("simulation_paused", round=self.current_round)

    def resume(self) -> None:
        """Resume the simulation."""
        self.status = SimulationStatus.RUNNING
        logger.info("simulation_resumed", round=self.current_round)

    def get_opinion_trajectory(self) -> dict[str, Any]:
        """Get opinion trajectories for all agents across rounds."""
        trajectories: dict[int, list[float]] = {
            a_id: [] for a_id in self.agents
        }

        for snapshot in self.opinion_history:
            for a_id in self.agents:
                trajectories[a_id].append(snapshot.get(a_id, 0.0))

        return {
            "topic": self.topic,
            "rounds": len(self.opinion_history),
            "agents": {
                a_id: {
                    "name": self.agents[a_id].name,
                    "archetype": self.agents[a_id].archetype,
                    "group_id": self.agents[a_id].group_id,
                    "trajectory": traj,
                }
                for a_id, traj in trajectories.items()
            },
        }

    def get_state_snapshot(self) -> dict[str, Any]:
        """Get current simulation state."""
        opinions = [a.opinion for a in self.agents.values()]
        return {
            "status": self.status.value,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "agent_count": len(self.agents),
            "avg_opinion": sum(opinions) / len(opinions) if opinions else 0,
            "opinion_range": (min(opinions), max(opinions)) if opinions else (0, 0),
            "total_actions": len(self.all_actions),
            "events_injected": len(self._event_history),
        }
