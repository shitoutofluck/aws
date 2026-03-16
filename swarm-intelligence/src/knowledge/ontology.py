"""LLM-driven ontology generation for knowledge graph construction.

Analyzes document content to determine what entity types and relationship
types should be extracted, following MiroFish's ontology-first approach.
"""

import json
import re
from typing import Any

import structlog

logger = structlog.get_logger()

ONTOLOGY_PROMPT = """You are an expert ontologist. Analyze the following document and design an ontology
for extracting a knowledge graph. The ontology should capture the key entity types and
relationship types present in the text.

DOCUMENT (first 2000 chars):
{document_preview}

Design an ontology with:
1. **Entity Types**: Categories of things mentioned (e.g., Person, Organization, Policy, Event).
   Each should have relevant attributes.
2. **Relationship Types**: How entities connect (e.g., works_at, opposes, supports).
   Each should specify valid source→target entity types.

Respond with valid JSON only:
{{
    "entity_types": [
        {{
            "name": "Person",
            "description": "An individual person mentioned in the text",
            "attributes": [
                {{"name": "role", "description": "Their role or title"}},
                {{"name": "affiliation", "description": "Organization or group affiliation"}}
            ]
        }}
    ],
    "edge_types": [
        {{
            "name": "works_at",
            "description": "Person works at an organization",
            "source_targets": [{{"source": "Person", "target": "Organization"}}]
        }}
    ]
}}"""


class OntologyGenerator:
    """Generates document-specific ontologies using LLM analysis."""

    async def generate(self, document_text: str, llm: Any) -> dict[str, Any]:
        """Generate an ontology from document content.

        Args:
            document_text: The source document text.
            llm: LangChain LLM instance.

        Returns:
            Ontology dict with entity_types and edge_types.
        """
        logger.info("generating_ontology", doc_length=len(document_text))

        preview = document_text[:2000]
        prompt = ONTOLOGY_PROMPT.format(document_preview=preview)

        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                ontology = json.loads(json_match.group())
            else:
                logger.warning("ontology_parse_failed_using_default")
                ontology = self._default_ontology()

        except (json.JSONDecodeError, Exception) as e:
            logger.warning("ontology_generation_failed", error=str(e))
            ontology = self._default_ontology()

        # Validate structure
        if "entity_types" not in ontology:
            ontology["entity_types"] = self._default_ontology()["entity_types"]
        if "edge_types" not in ontology:
            ontology["edge_types"] = self._default_ontology()["edge_types"]

        logger.info(
            "ontology_generated",
            entity_types=len(ontology["entity_types"]),
            edge_types=len(ontology["edge_types"]),
        )
        return ontology

    @staticmethod
    def _default_ontology() -> dict[str, Any]:
        """Fallback ontology for general-purpose extraction."""
        return {
            "entity_types": [
                {
                    "name": "Person",
                    "description": "An individual person",
                    "attributes": [
                        {"name": "role", "description": "Their role or title"},
                        {"name": "affiliation", "description": "Organization affiliation"},
                    ],
                },
                {
                    "name": "Organization",
                    "description": "A company, institution, or group",
                    "attributes": [
                        {"name": "sector", "description": "Industry or sector"},
                    ],
                },
                {
                    "name": "Place",
                    "description": "A geographic location",
                    "attributes": [
                        {"name": "region", "description": "Geographic region"},
                    ],
                },
                {
                    "name": "Event",
                    "description": "A notable event or occurrence",
                    "attributes": [
                        {"name": "date", "description": "When it occurred"},
                    ],
                },
                {
                    "name": "Policy",
                    "description": "A policy, regulation, or decision",
                    "attributes": [
                        {"name": "domain", "description": "Policy domain"},
                    ],
                },
                {
                    "name": "Concept",
                    "description": "An abstract concept or topic",
                    "attributes": [],
                },
            ],
            "edge_types": [
                {
                    "name": "works_at",
                    "description": "Person works at organization",
                    "source_targets": [{"source": "Person", "target": "Organization"}],
                },
                {
                    "name": "located_in",
                    "description": "Entity is located in a place",
                    "source_targets": [{"source": "Organization", "target": "Place"}],
                },
                {
                    "name": "supports",
                    "description": "Entity supports another entity or policy",
                    "source_targets": [
                        {"source": "Person", "target": "Policy"},
                        {"source": "Organization", "target": "Policy"},
                    ],
                },
                {
                    "name": "opposes",
                    "description": "Entity opposes another entity or policy",
                    "source_targets": [
                        {"source": "Person", "target": "Policy"},
                        {"source": "Organization", "target": "Policy"},
                    ],
                },
                {
                    "name": "member_of",
                    "description": "Person is member of organization or group",
                    "source_targets": [{"source": "Person", "target": "Organization"}],
                },
                {
                    "name": "affects",
                    "description": "One entity affects another",
                    "source_targets": [
                        {"source": "Event", "target": "Person"},
                        {"source": "Policy", "target": "Organization"},
                    ],
                },
            ],
        }
