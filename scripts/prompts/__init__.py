"""Structured prompts for MedQuery classification.

Provides modular prompt components for:
- Medical domain boundary detection
- Entity extraction with UMLS mapping
- Intent classification
- Relationship extraction
- Query decomposition
"""

from .schemas import OUTPUT_SCHEMA, ENTITY_TYPES, RELATIONSHIP_TYPES, INTENT_TYPES
from .master import MasterPrompt

__all__ = [
    "MasterPrompt",
    "OUTPUT_SCHEMA",
    "ENTITY_TYPES",
    "RELATIONSHIP_TYPES",
    "INTENT_TYPES",
]
