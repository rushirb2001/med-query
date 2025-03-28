"""Query generators for MedQuery test dataset.

LLM-assisted generation of 10,000 diverse test queries with:
- Medical vs non-medical classification
- Intent classification (conceptual, procedural, relationship, lookup)
- Entity extraction with UMLS CUI mapping
- Relationship extraction
- Query decomposition for complex queries
"""

from .base import QueryGenerator, GeneratedQuery
from .templates import QUERY_TEMPLATES
from .validate import validate_query, deduplicate_queries

__all__ = [
    "QueryGenerator",
    "GeneratedQuery",
    "QUERY_TEMPLATES",
    "validate_query",
    "deduplicate_queries",
]
