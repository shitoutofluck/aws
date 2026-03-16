"""Report agent for simulation analysis and output generation.

Analyzes simulation results and generates structured reports, predictions,
and summaries. Inspired by MiroFish's ReportAgent with tool access and
reflection loops.
"""

import json
import re
from typing import Any

import structlog

from src.simulation.engine import RoundSummary, SimulationEngine

logger = structlog.get_logger()

REPORT_PROMPT = """You are a senior analyst reviewing a social simulation. Your task is to generate
a comprehensive report analyzing the simulation dynamics and predicting outcomes.

SIMULATION OVERVIEW:
- Topic: {topic}
- Rounds completed: {rounds}
- Total agents: {agent_count}
- Total actions taken: {total_actions}

OPINION TRAJECTORY:
- Starting average opinion: {start_opinion:.3f}
- Final average opinion: {end_opinion:.3f}
- Opinion standard deviation: {opinion_std:.3f}
- Direction of shift: {shift_direction}

GROUP DYNAMICS:
{group_dynamics}

KEY EVENTS INJECTED:
{events}

ROUND-BY-ROUND SUMMARY:
{round_summaries}

NOTABLE AGENT BEHAVIORS:
{notable_behaviors}

Generate a structured analysis report with:
1. **Executive Summary**: 2-3 sentences on the overall simulation outcome
2. **Key Findings**: Bullet points of the most important observations
3. **Opinion Dynamics**: How opinions evolved and why
4. **Group Behavior**: Emergent group behaviors observed
5. **Event Impact**: How injected events affected the simulation
6. **Predictions**: What would likely happen if the simulation continued
7. **Confidence Level**: How confident are you in these predictions (low/medium/high)

Be analytical and specific. Reference actual data from the simulation."""


class ReportAgent:
    """Generates analytical reports from simulation results."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        logger.info("report_agent_initialized")

    async def generate_report(self, engine: SimulationEngine) -> dict[str, Any]:
        """Generate a full analysis report from simulation results.

        Args:
            engine: Completed SimulationEngine instance.

        Returns:
            Dict with structured report sections.
        """
        logger.info("generating_report", rounds=len(engine.round_summaries))

        # Compute analytics
        analytics = self._compute_analytics(engine)

        # Generate LLM report
        prompt = REPORT_PROMPT.format(**analytics)

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error("report_generation_failed", error=str(e))
            content = self._fallback_report(analytics)

        report = {
            "report_text": content,
            "analytics": analytics,
            "opinion_trajectory": engine.get_opinion_trajectory(),
            "state_snapshot": engine.get_state_snapshot(),
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        }

        logger.info("report_generated")
        return report

    def _compute_analytics(self, engine: SimulationEngine) -> dict[str, Any]:
        """Compute analytics from simulation data."""
        summaries = engine.round_summaries
        agents = engine.agents

        # Opinion trajectory
        if engine.opinion_history:
            start_opinions = list(engine.opinion_history[0].values())
            end_opinions = list(engine.opinion_history[-1].values())
            start_avg = sum(start_opinions) / len(start_opinions)
            end_avg = sum(end_opinions) / len(end_opinions)
            end_std = (sum((o - end_avg) ** 2 for o in end_opinions) / len(end_opinions)) ** 0.5
        else:
            start_avg = end_avg = end_std = 0.0

        shift = end_avg - start_avg
        if abs(shift) < 0.05:
            shift_direction = "Stable (minimal change)"
        elif shift > 0:
            shift_direction = f"Positive shift (+{shift:.3f})"
        else:
            shift_direction = f"Negative shift ({shift:.3f})"

        # Group dynamics
        group_opinions: dict[int, list[float]] = {}
        for agent in agents.values():
            gid = agent.group_id or 0
            group_opinions.setdefault(gid, []).append(agent.opinion)

        group_dynamics_lines = []
        for gid, opinions in sorted(group_opinions.items()):
            avg = sum(opinions) / len(opinions)
            group_dynamics_lines.append(
                f"Group {gid}: {len(opinions)} agents, avg opinion: {avg:.3f}"
            )
        group_dynamics = "\n".join(group_dynamics_lines)

        # Events
        events = "\n".join(
            f"- Round {r}: {evt}" for r, evt in engine._event_history
        ) if engine._event_history else "No events were injected."

        # Round summaries
        round_summary_lines = []
        for s in summaries:
            round_summary_lines.append(
                f"Round {s.round_num}: {s.active_agents} active, "
                f"avg opinion {s.avg_opinion:.3f}, {len(s.actions)} actions"
            )
        round_summaries = "\n".join(round_summary_lines)

        # Notable behaviors
        notable = self._find_notable_behaviors(engine)

        return {
            "topic": engine.topic,
            "rounds": len(summaries),
            "agent_count": len(agents),
            "total_actions": len(engine.all_actions),
            "start_opinion": start_avg,
            "end_opinion": end_avg,
            "opinion_std": end_std,
            "shift_direction": shift_direction,
            "group_dynamics": group_dynamics,
            "events": events,
            "round_summaries": round_summaries,
            "notable_behaviors": notable,
        }

    def _find_notable_behaviors(self, engine: SimulationEngine) -> str:
        """Identify notable agent behaviors from the simulation."""
        if not engine.opinion_history or len(engine.opinion_history) < 2:
            return "Insufficient data for behavior analysis."

        lines = []
        start = engine.opinion_history[0]
        end = engine.opinion_history[-1]

        # Find biggest opinion shifters
        shifts = []
        for agent_id in engine.agents:
            s = start.get(agent_id, 0)
            e = end.get(agent_id, 0)
            shifts.append((agent_id, e - s, e))

        shifts.sort(key=lambda x: abs(x[1]), reverse=True)

        for agent_id, shift, final in shifts[:5]:
            agent = engine.agents[agent_id]
            lines.append(
                f"- {agent.name} ({agent.archetype}): shifted {shift:+.3f} to {final:.3f}"
            )

        # Find most active agents
        action_counts: dict[int, int] = {}
        for action in engine.all_actions:
            action_counts[action.agent_id] = action_counts.get(action.agent_id, 0) + 1

        most_active = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for agent_id, count in most_active:
            agent = engine.agents[agent_id]
            lines.append(f"- {agent.name}: most active with {count} actions")

        return "\n".join(lines) if lines else "No notable behaviors detected."

    def _fallback_report(self, analytics: dict[str, Any]) -> str:
        """Generate a simple report when LLM fails."""
        return (
            f"## Simulation Report: {analytics['topic']}\n\n"
            f"**Rounds completed:** {analytics['rounds']}\n"
            f"**Agents:** {analytics['agent_count']}\n"
            f"**Total actions:** {analytics['total_actions']}\n\n"
            f"### Opinion Dynamics\n"
            f"- Start: {analytics['start_opinion']:.3f}\n"
            f"- End: {analytics['end_opinion']:.3f}\n"
            f"- Shift: {analytics['shift_direction']}\n\n"
            f"### Group Dynamics\n{analytics['group_dynamics']}\n\n"
            f"### Events\n{analytics['events']}\n"
        )
