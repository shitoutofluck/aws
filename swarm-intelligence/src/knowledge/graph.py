"""Document processing and local knowledge graph using NetworkX.

Replaces Zep Cloud dependency with a fully local GraphRAG implementation.
Extracts entities, relationships, and contexts from documents using LLM,
then builds a queryable knowledge graph with NetworkX.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import networkx as nx
import structlog

logger = structlog.get_logger()


@dataclass
class Entity:
    """A node in the knowledge graph."""

    uuid: str
    name: str
    entity_type: str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }


@dataclass
class Relationship:
    """An edge in the knowledge graph."""

    uuid: str
    source_uuid: str
    target_uuid: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "source_uuid": self.source_uuid,
            "target_uuid": self.target_uuid,
            "relation_type": self.relation_type,
            "description": self.description,
            "weight": self.weight,
            "attributes": self.attributes,
        }


@dataclass
class GraphInfo:
    """Summary information about the knowledge graph."""

    graph_id: str
    node_count: int
    edge_count: int
    entity_types: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class TextProcessor:
    """Splits documents into chunks for processing."""

    @staticmethod
    def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1

            chunks.append(text[start:end].strip())
            start = end - overlap

        return [c for c in chunks if c]


EXTRACTION_PROMPT = """You are an expert entity and relationship extractor. Analyze the following text and extract:

1. **Entities**: People, organizations, places, concepts, events, policies, groups
2. **Relationships**: How entities relate to each other

For each entity, provide:
- name: Clear entity name
- type: One of [Person, Organization, Place, Concept, Event, Policy, Group, Technology]
- description: Brief description based on the text

For each relationship, provide:
- source: Source entity name
- target: Target entity name
- relation: Relationship type (e.g., "works_at", "located_in", "affects", "member_of", "opposes", "supports")
- description: Brief description

TEXT:
{text}

Respond with valid JSON only:
{{
    "entities": [
        {{"name": "...", "type": "...", "description": "..."}}
    ],
    "relationships": [
        {{"source": "...", "target": "...", "relation": "...", "description": "..."}}
    ]
}}"""


class KnowledgeGraph:
    """Local knowledge graph built from documents using LLM + NetworkX.

    This replaces MiroFish's Zep Cloud dependency with a fully local,
    cost-free implementation that works offline with Ollama.
    """

    def __init__(self, graph_id: str | None = None) -> None:
        self.graph_id = graph_id or f"kg_{uuid.uuid4().hex[:12]}"
        self.graph = nx.DiGraph()
        self.entities: dict[str, Entity] = {}
        self.relationships: dict[str, Relationship] = {}
        self._entity_name_to_uuid: dict[str, str] = {}
        logger.info("knowledge_graph_created", graph_id=self.graph_id)

    async def build_from_document(
        self,
        text: str,
        llm: Any,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> GraphInfo:
        """Build knowledge graph from a document using LLM extraction.

        Args:
            text: Document text to process.
            llm: LangChain LLM instance for entity extraction.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.

        Returns:
            GraphInfo with summary statistics.
        """
        logger.info("building_graph_from_document", text_length=len(text))

        # Split text into chunks
        chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
        logger.info("text_split", chunk_count=len(chunks))

        # Extract entities and relationships from each chunk
        for i, chunk in enumerate(chunks):
            logger.debug("processing_chunk", chunk_index=i, total=len(chunks))
            try:
                await self._extract_from_chunk(chunk, llm)
            except Exception as e:
                logger.warning("chunk_extraction_failed", chunk_index=i, error=str(e))
                continue

        # Merge duplicate entities
        self._merge_duplicates()

        info = self.get_info()
        logger.info(
            "graph_built",
            nodes=info.node_count,
            edges=info.edge_count,
            types=info.entity_types,
        )
        return info

    async def _extract_from_chunk(self, chunk: str, llm: Any) -> None:
        """Extract entities and relationships from a text chunk."""
        prompt = EXTRACTION_PROMPT.format(text=chunk)

        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Parse JSON from response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if not json_match:
                return

            data = json.loads(json_match.group())
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("extraction_parse_failed", error=str(e))
            return

        # Add entities
        for ent_data in data.get("entities", []):
            name = ent_data.get("name", "").strip()
            if not name:
                continue
            self._add_entity(
                name=name,
                entity_type=ent_data.get("type", "Concept"),
                description=ent_data.get("description", ""),
            )

        # Add relationships
        for rel_data in data.get("relationships", []):
            source = rel_data.get("source", "").strip()
            target = rel_data.get("target", "").strip()
            if not source or not target:
                continue
            self._add_relationship(
                source_name=source,
                target_name=target,
                relation_type=rel_data.get("relation", "related_to"),
                description=rel_data.get("description", ""),
            )

    def _add_entity(self, name: str, entity_type: str, description: str = "") -> Entity:
        """Add or update an entity in the graph."""
        name_lower = name.lower()

        if name_lower in self._entity_name_to_uuid:
            # Update existing entity description if richer
            existing = self.entities[self._entity_name_to_uuid[name_lower]]
            if len(description) > len(existing.description):
                existing.description = description
            return existing

        entity = Entity(
            uuid=f"ent_{uuid.uuid4().hex[:10]}",
            name=name,
            entity_type=entity_type,
            description=description,
        )

        self.entities[entity.uuid] = entity
        self._entity_name_to_uuid[name_lower] = entity.uuid
        self.graph.add_node(
            entity.uuid,
            name=name,
            entity_type=entity_type,
            description=description,
        )
        return entity

    def _add_relationship(
        self,
        source_name: str,
        target_name: str,
        relation_type: str,
        description: str = "",
    ) -> Relationship | None:
        """Add a relationship between two entities."""
        source_uuid = self._entity_name_to_uuid.get(source_name.lower())
        target_uuid = self._entity_name_to_uuid.get(target_name.lower())

        # Create entities if they don't exist
        if not source_uuid:
            entity = self._add_entity(source_name, "Concept")
            source_uuid = entity.uuid
        if not target_uuid:
            entity = self._add_entity(target_name, "Concept")
            target_uuid = entity.uuid

        rel = Relationship(
            uuid=f"rel_{uuid.uuid4().hex[:10]}",
            source_uuid=source_uuid,
            target_uuid=target_uuid,
            relation_type=relation_type,
            description=description,
        )

        self.relationships[rel.uuid] = rel
        self.graph.add_edge(
            source_uuid,
            target_uuid,
            relation_type=relation_type,
            description=description,
            uuid=rel.uuid,
        )
        return rel

    def _merge_duplicates(self) -> None:
        """Merge duplicate entities with similar names."""
        # Simple dedup already handled by name-based lookup in _add_entity
        pass

    def get_info(self) -> GraphInfo:
        """Get graph summary information."""
        entity_types = list({e.entity_type for e in self.entities.values()})
        return GraphInfo(
            graph_id=self.graph_id,
            node_count=len(self.entities),
            edge_count=len(self.relationships),
            entity_types=entity_types,
        )

    def get_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_person_entities(self) -> list[Entity]:
        """Get all person-type entities for agent generation."""
        person_types = {"Person", "Group", "Organization"}
        return [e for e in self.entities.values() if e.entity_type in person_types]

    def get_neighbors(self, entity_uuid: str) -> list[tuple[Entity, Relationship]]:
        """Get all neighboring entities and their relationships."""
        neighbors = []
        for _, target_uuid, data in self.graph.out_edges(entity_uuid, data=True):
            if target_uuid in self.entities:
                rel_uuid = data.get("uuid")
                rel = self.relationships.get(rel_uuid) if rel_uuid else None
                neighbors.append((self.entities[target_uuid], rel))
        for source_uuid, _, data in self.graph.in_edges(entity_uuid, data=True):
            if source_uuid in self.entities:
                rel_uuid = data.get("uuid")
                rel = self.relationships.get(rel_uuid) if rel_uuid else None
                neighbors.append((self.entities[source_uuid], rel))
        return neighbors

    def get_entity_context(self, entity_uuid: str) -> str:
        """Get rich context string for an entity (for agent biography generation)."""
        entity = self.entities.get(entity_uuid)
        if not entity:
            return ""

        lines = [f"Entity: {entity.name} (Type: {entity.entity_type})"]
        if entity.description:
            lines.append(f"Description: {entity.description}")

        neighbors = self.get_neighbors(entity_uuid)
        if neighbors:
            lines.append("\nConnections:")
            for neighbor, rel in neighbors:
                rel_desc = f" ({rel.relation_type}: {rel.description})" if rel else ""
                lines.append(f"  - {neighbor.name} [{neighbor.entity_type}]{rel_desc}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full graph to dictionary."""
        return {
            "graph_id": self.graph_id,
            "entities": [e.to_dict() for e in self.entities.values()],
            "relationships": [r.to_dict() for r in self.relationships.values()],
        }

    def save(self, filepath: str) -> None:
        """Save graph to JSON file."""
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("graph_saved", path=filepath)

    @classmethod
    def load(cls, filepath: str) -> "KnowledgeGraph":
        """Load graph from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        kg = cls(graph_id=data["graph_id"])

        for ent_data in data["entities"]:
            entity = Entity(**ent_data)
            kg.entities[entity.uuid] = entity
            kg._entity_name_to_uuid[entity.name.lower()] = entity.uuid
            kg.graph.add_node(
                entity.uuid,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
            )

        for rel_data in data["relationships"]:
            rel = Relationship(**rel_data)
            kg.relationships[rel.uuid] = rel
            kg.graph.add_edge(
                rel.source_uuid,
                rel.target_uuid,
                relation_type=rel.relation_type,
                description=rel.description,
                uuid=rel.uuid,
            )

        logger.info("graph_loaded", path=filepath, nodes=len(kg.entities))
        return kg
