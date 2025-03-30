"""Pydantic schemas for model output validation.

Defines the expected output structure from LLM query classification,
including entities, relationships, intents, and decomposition.
"""

from typing import Literal
from pydantic import BaseModel, Field


# Type definitions
EntityType = Literal["condition", "procedure", "anatomy", "process", "concept", "medication"]
RelationshipType = Literal["affects", "causes", "treats", "indicates", "has_property", "compared_to", "part_of", "precedes"]
IntentType = Literal["conceptual", "procedural", "relationship", "lookup"]
RetrievalStrategy = Literal["vector_search", "graph_traversal", "hybrid_search", "metadata_lookup"]


class Entity(BaseModel):
    """Medical entity with optional UMLS CUI."""

    text: str = Field(..., description="Entity text as it appears in query")
    type: EntityType = Field(..., description="Entity type classification")
    cui: str | None = Field(None, description="UMLS Concept Unique Identifier")

    def __hash__(self):
        return hash((self.text.lower(), self.type))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.text.lower() == other.text.lower() and self.type == other.type


class Relationship(BaseModel):
    """Relationship between two entities."""

    source: str = Field(..., description="Source entity text")
    target: str = Field(..., description="Target entity text")
    type: RelationshipType = Field(..., description="Relationship type")

    def __hash__(self):
        return hash((self.source.lower(), self.target.lower(), self.type))

    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return (
            self.source.lower() == other.source.lower()
            and self.target.lower() == other.target.lower()
            and self.type == other.type
        )


class SubQuery(BaseModel):
    """Decomposed sub-query for complex queries."""

    query: str = Field(..., description="Sub-query text")
    intent: str = Field(..., description="Sub-query intent")


class QueryAnalysis(BaseModel):
    """Complete analysis output from the model."""

    is_medical: bool = Field(..., description="Whether query is medical domain")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for medical classification",
    )
    primary_intent: IntentType | None = Field(
        None,
        description="Primary query intent",
    )
    secondary_intent: str | None = Field(
        None,
        description="Optional secondary intent",
    )
    entities: list[Entity] = Field(
        default_factory=list,
        description="Extracted entities",
    )
    relationships: list[Relationship] = Field(
        default_factory=list,
        description="Extracted relationships",
    )
    decomposition: list[SubQuery] = Field(
        default_factory=list,
        description="Sub-queries for complex queries",
    )
    retrieval_strategy: list[RetrievalStrategy] = Field(
        default_factory=list,
        description="Recommended retrieval strategies",
    )

    model_config = {"extra": "ignore"}  # Ignore extra fields in lenient mode


class ValidationResult(BaseModel):
    """Result of validating model output."""

    valid: bool = Field(..., description="Whether output is valid")
    parsed: QueryAnalysis | None = Field(
        None,
        description="Parsed output if successful",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Validation errors",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Validation warnings (non-fatal)",
    )
    raw_output: str = Field(..., description="Original raw output")
    parse_time_ms: float = Field(
        0.0,
        description="Time to parse and validate (ms)",
    )


class ExpectedOutput(BaseModel):
    """Expected output from test dataset for comparison."""

    is_medical: bool
    confidence: float = Field(ge=0.0, le=1.0)
    primary_intent: IntentType | None = None
    secondary_intent: str | None = None
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    decomposition: list[SubQuery] = Field(default_factory=list)
    retrieval_strategy: list[RetrievalStrategy] = Field(default_factory=list)


class TestQuery(BaseModel):
    """Test query with expected output for evaluation."""

    id: str
    query: str
    level: str  # L1-L5
    category: str
    expected: ExpectedOutput
