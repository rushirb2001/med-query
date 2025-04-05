"""MedQuery - LangGraph-based query routing and orchestration for medical textbook retrieval."""

__version__ = "0.1.0"

# Logging utilities for programmatic use
from .logging import setup_logging, get_logger, verbose, enable_debug

# Type definitions
from .types import (
    EntityType,
    RelationshipType,
    IntentType,
    RetrievalStrategy,
    ENTITY_TYPES,
    RELATIONSHIP_TYPES,
    INTENT_TYPES,
    OUTPUT_SCHEMA,
    OUTPUT_SCHEMA_STR,
)

__all__ = [
    "__version__",
    # Logging
    "setup_logging",
    "get_logger",
    "verbose",
    "enable_debug",
    # Types
    "EntityType",
    "RelationshipType",
    "IntentType",
    "RetrievalStrategy",
    "ENTITY_TYPES",
    "RELATIONSHIP_TYPES",
    "INTENT_TYPES",
    "OUTPUT_SCHEMA",
    "OUTPUT_SCHEMA_STR",
]
