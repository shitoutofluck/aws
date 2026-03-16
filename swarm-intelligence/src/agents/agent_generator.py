"""Dynamic agent generation from knowledge graph entities.

Transforms knowledge graph nodes into autonomous agents with unique
biographies, personalities, and social connections. Inspired by MiroFish's
OasisProfileGenerator which generates agents from Zep graph entities.
"""

import json
import random
import re
from typing import Any

import structlog

from src.knowledge.graph import Entity, KnowledgeGraph

from .agent_profile import ARCHETYPES, AgentProfile

logger = structlog.get_logger()

BIOGRAPHY_PROMPT = """You are creating a detailed character profile for an agent in a social simulation.
Based on the following entity information from a knowledge graph, generate a rich biography and persona.

ENTITY INFORMATION:
{entity_context}

Generate a JSON profile with these fields:
{{
    "bio": "A 2-3 sentence biography capturing who this agent is, their background, and motivations",
    "persona": "A detailed 4-6 sentence persona description including personality traits, values, how they form opinions, and how they interact socially",
    "age": <integer 18-80 or null>,
    "gender": "<male/female/nonbinary or null>",
    "profession": "<their profession or role>",
    "interested_topics": ["topic1", "topic2", "topic3"]
}}

Make the persona vivid and specific. The agent should feel like a real individual with
clear behavioral tendencies that will emerge during social simulation."""


class AgentGenerator:
    """Generates diverse autonomous agents from knowledge graph entities.

    This is the key transformation: knowledge graph nodes become simulation
    participants with unique personalities, social ties, and behavioral logic.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self.kg = knowledge_graph
        logger.info("agent_generator_initialized", graph_id=knowledge_graph.graph_id)

    async def generate_agents(
        self,
        llm: Any,
        max_agents: int = 50,
        use_llm_for_bios: bool = True,
        min_agents: int = 5,
    ) -> list[AgentProfile]:
        """Generate agent profiles from knowledge graph entities.

        Args:
            llm: LangChain LLM for biography generation.
            max_agents: Maximum number of agents to create.
            use_llm_for_bios: Whether to use LLM for rich biographies.
            min_agents: Minimum agents; synthetic agents created if graph is sparse.

        Returns:
            List of AgentProfile instances.
        """
        logger.info(
            "generating_agents",
            max_agents=max_agents,
            use_llm=use_llm_for_bios,
        )

        # Get person-like entities from graph
        person_entities = self.kg.get_person_entities()
        all_entities = list(self.kg.entities.values())

        # Select entities for agent generation
        selected = person_entities[:max_agents]

        # If not enough person entities, include other types
        if len(selected) < min_agents:
            remaining = [e for e in all_entities if e not in selected]
            random.shuffle(remaining)
            selected.extend(remaining[: min_agents - len(selected)])

        logger.info("entities_selected", count=len(selected))

        # Generate profiles
        profiles: list[AgentProfile] = []
        for idx, entity in enumerate(selected):
            try:
                profile = await self._generate_profile(
                    agent_id=idx,
                    entity=entity,
                    llm=llm if use_llm_for_bios else None,
                )
                profiles.append(profile)
                logger.debug("agent_generated", agent_id=idx, name=profile.name)
            except Exception as e:
                logger.warning(
                    "agent_generation_failed",
                    agent_id=idx,
                    entity=entity.name,
                    error=str(e),
                )
                # Create fallback profile
                profiles.append(self._fallback_profile(idx, entity))

        # Generate social connections between agents
        self._generate_social_network(profiles)

        # Assign group memberships
        self._assign_groups(profiles)

        logger.info("agents_generated", count=len(profiles))
        return profiles

    async def _generate_profile(
        self,
        agent_id: int,
        entity: Entity,
        llm: Any | None = None,
    ) -> AgentProfile:
        """Generate a single agent profile from an entity."""
        # Get entity context from knowledge graph
        context = self.kg.get_entity_context(entity.uuid)

        # Generate random personality traits
        personality = AgentProfile.random_personality()

        profile = AgentProfile(
            agent_id=agent_id,
            name=entity.name,
            entity_type=entity.entity_type,
            source_entity_uuid=entity.uuid,
            source_entity_type=entity.entity_type,
            memory_key=f"agent_{agent_id}_{entity.uuid[:8]}",
            **personality,
        )

        # Use LLM for rich biography if available
        if llm and context:
            try:
                bio_data = await self._generate_biography(context, llm)
                profile.bio = bio_data.get("bio", entity.description)
                profile.persona = bio_data.get("persona", "")
                profile.age = bio_data.get("age")
                profile.gender = bio_data.get("gender")
                profile.profession = bio_data.get("profession")
                profile.interested_topics = bio_data.get("interested_topics", [])
            except Exception as e:
                logger.warning("biography_generation_failed", error=str(e))
                profile.bio = entity.description
        else:
            profile.bio = entity.description

        return profile

    async def _generate_biography(self, entity_context: str, llm: Any) -> dict[str, Any]:
        """Use LLM to generate a rich biography from entity context."""
        prompt = BIOGRAPHY_PROMPT.format(entity_context=entity_context)

        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json.loads(json_match.group())

        return {"bio": entity_context, "persona": "", "interested_topics": []}

    def _fallback_profile(self, agent_id: int, entity: Entity) -> AgentProfile:
        """Create a simple profile without LLM."""
        personality = AgentProfile.random_personality()
        return AgentProfile(
            agent_id=agent_id,
            name=entity.name,
            entity_type=entity.entity_type,
            bio=entity.description or f"Agent representing {entity.name}",
            persona=f"A {personality['archetype'].lower()} personality involved in {entity.entity_type.lower()} matters.",
            source_entity_uuid=entity.uuid,
            source_entity_type=entity.entity_type,
            memory_key=f"agent_{agent_id}_{entity.uuid[:8]}",
            **personality,
        )

    def _generate_social_network(self, profiles: list[AgentProfile]) -> None:
        """Generate social connections based on knowledge graph relationships.

        Agents connected in the knowledge graph get social ties. Additional
        random connections create small-world network properties.
        """
        if len(profiles) < 2:
            return

        # Map entity UUIDs to agent IDs
        uuid_to_agent: dict[str, int] = {}
        for p in profiles:
            if p.source_entity_uuid:
                uuid_to_agent[p.source_entity_uuid] = p.agent_id

        # Create connections from graph relationships
        for rel in self.kg.relationships.values():
            src_agent = uuid_to_agent.get(rel.source_uuid)
            tgt_agent = uuid_to_agent.get(rel.target_uuid)
            if src_agent is not None and tgt_agent is not None and src_agent != tgt_agent:
                if tgt_agent not in profiles[src_agent].connections:
                    profiles[src_agent].connections.append(tgt_agent)
                if src_agent not in profiles[tgt_agent].connections:
                    profiles[tgt_agent].connections.append(src_agent)

        # Add random connections for small-world effect
        agent_ids = [p.agent_id for p in profiles]
        for profile in profiles:
            # Each agent gets 1-3 random connections beyond graph-based ones
            num_random = random.randint(1, min(3, len(agent_ids) - 1))
            candidates = [a for a in agent_ids if a != profile.agent_id and a not in profile.connections]
            if candidates:
                random_connections = random.sample(candidates, min(num_random, len(candidates)))
                profile.connections.extend(random_connections)

        logger.info(
            "social_network_generated",
            avg_connections=sum(len(p.connections) for p in profiles) / max(len(profiles), 1),
        )

    def _assign_groups(self, profiles: list[AgentProfile], num_groups: int | None = None) -> None:
        """Assign agents to groups/clusters using simple community detection.

        Groups enable emergent behaviors like opinion polarization and
        herd effects within the simulation.
        """
        if not profiles:
            return

        if num_groups is None:
            num_groups = max(2, len(profiles) // 5)

        # Simple assignment: use archetype clustering + random
        archetype_groups: dict[str, int] = {}
        group_counter = 0

        for profile in profiles:
            arch = profile.archetype
            if arch not in archetype_groups:
                archetype_groups[arch] = group_counter % num_groups
                group_counter += 1
            profile.group_id = archetype_groups[arch]

        logger.info("groups_assigned", num_groups=num_groups)
