"""Output schemas and type definitions for MedQuery classification.

Defines the structured output format and valid values for:
- Entity types
- Relationship types
- Intent types
- Full classification output
"""

# =============================================================================
# ENTITY TYPES
# =============================================================================

ENTITY_TYPES = {
    "condition": "Diseases, syndromes, injuries (e.g., sepsis, TBI, fracture)",
    "procedure": "Surgical/medical procedures (e.g., thoracotomy, intubation)",
    "anatomy": "Body parts, organs, structures (e.g., liver, femoral artery)",
    "process": "Physiological processes (e.g., coagulation, inflammation)",
    "concept": "Abstract medical concepts (e.g., triage, hemostasis)",
    "medication": "Drugs, treatments (e.g., epinephrine, TXA, blood products)",
}


# =============================================================================
# RELATIONSHIP TYPES
# =============================================================================

RELATIONSHIP_TYPES = {
    "affects": "X influences/changes Y",
    "causes": "X leads to/produces Y",
    "treats": "X is used to treat Y",
    "indicates": "X is a sign/symptom of Y",
    "has_property": "X has characteristic Y",
    "compared_to": "X is being compared with Y",
    "part_of": "X is a component of Y",
    "precedes": "X comes before Y (in time or procedure)",
}


# =============================================================================
# INTENT TYPES
# =============================================================================

INTENT_TYPES = {
    "conceptual": {
        "description": "What is X? Definitions, explanations, pathophysiology",
        "patterns": ["what is", "explain", "define", "describe", "meaning of"],
        "retrieval": "vector_search",
    },
    "procedural": {
        "description": "How to do X? Steps, techniques, procedures",
        "patterns": ["how to", "steps for", "technique", "perform", "describe procedure"],
        "retrieval": "hybrid_search",
    },
    "relationship": {
        "description": "How does X affect Y? Comparisons, connections, effects",
        "patterns": ["affect", "effect", "between", "compare", "vs", "relationship", "connection"],
        "retrieval": "graph_traversal",
    },
    "lookup": {
        "description": "Find chapter/section/reference for X",
        "patterns": ["chapter", "section", "page", "find", "reference", "where is"],
        "retrieval": "metadata_lookup",
    },
}


# =============================================================================
# OUTPUT SCHEMA
# =============================================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["medical", "intent"],
    "properties": {
        "medical": {
            "type": "object",
            "required": ["value", "confidence"],
            "properties": {
                "value": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "intent": {
            "type": "object",
            "required": ["primary"],
            "properties": {
                "primary": {"enum": ["conceptual", "procedural", "relationship", "lookup", None]},
                "secondary": {"enum": ["conceptual", "procedural", "relationship", "lookup", None]},
            },
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text", "type"],
                "properties": {
                    "text": {"type": "string"},
                    "type": {"enum": list(ENTITY_TYPES.keys())},
                    "cui": {"type": ["string", "null"]},
                },
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["source", "target", "type"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"enum": list(RELATIONSHIP_TYPES.keys())},
                },
            },
        },
        "decompose": {"type": "boolean"},
        "sub_queries": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "required": ["query", "intent"],
                "properties": {
                    "query": {"type": "string"},
                    "intent": {"enum": ["conceptual", "procedural", "relationship", "lookup"]},
                },
            },
        },
        "retrieval_hint": {
            "enum": ["vector_search", "graph_traversal", "hybrid_search", "metadata_lookup", "graph_then_vector"],
        },
    },
}


# =============================================================================
# SCHEMA AS STRING FOR PROMPTS
# =============================================================================

OUTPUT_SCHEMA_STR = """{
  "medical": {"value": true|false, "confidence": 0.0-1.0},
  "intent": {"primary": "conceptual"|"procedural"|"relationship"|"lookup"|null, "secondary": ...},
  "entities": [{"text": "...", "type": "condition|procedure|anatomy|process|concept|medication", "cui": null}],
  "relationships": [{"source": "...", "target": "...", "type": "affects|causes|treats|indicates|compared_to"}],
  "decompose": true|false,
  "sub_queries": [{"query": "...", "intent": "..."}] | null,
  "retrieval_hint": "vector_search"|"graph_traversal"|"hybrid_search"|"metadata_lookup"|"graph_then_vector"
}"""


# =============================================================================
# EXAMPLE OUTPUTS
# =============================================================================

EXAMPLE_OUTPUTS = {
    "simple_conceptual": {
        "query": "What is hemorrhagic shock?",
        "output": {
            "medical": {"value": True, "confidence": 0.98},
            "intent": {"primary": "conceptual", "secondary": None},
            "entities": [{"text": "hemorrhagic shock", "type": "condition", "cui": "C0038454"}],
            "relationships": [],
            "decompose": False,
            "sub_queries": None,
            "retrieval_hint": "vector_search",
        },
    },
    "simple_procedural": {
        "query": "How to perform a thoracotomy?",
        "output": {
            "medical": {"value": True, "confidence": 0.99},
            "intent": {"primary": "procedural", "secondary": None},
            "entities": [{"text": "thoracotomy", "type": "procedure", "cui": "C0039991"}],
            "relationships": [],
            "decompose": False,
            "sub_queries": None,
            "retrieval_hint": "hybrid_search",
        },
    },
    "relationship": {
        "query": "How does hemorrhagic shock affect the coagulation cascade?",
        "output": {
            "medical": {"value": True, "confidence": 0.97},
            "intent": {"primary": "relationship", "secondary": None},
            "entities": [
                {"text": "hemorrhagic shock", "type": "condition", "cui": "C0038454"},
                {"text": "coagulation cascade", "type": "process", "cui": "C0005778"},
            ],
            "relationships": [
                {"source": "hemorrhagic shock", "target": "coagulation cascade", "type": "affects"},
            ],
            "decompose": False,
            "sub_queries": None,
            "retrieval_hint": "graph_then_vector",
        },
    },
    "complex_decomposition": {
        "query": "What is damage control surgery and what are the steps for temporary abdominal closure?",
        "output": {
            "medical": {"value": True, "confidence": 0.95},
            "intent": {"primary": "conceptual", "secondary": "procedural"},
            "entities": [
                {"text": "damage control surgery", "type": "procedure", "cui": None},
                {"text": "temporary abdominal closure", "type": "procedure", "cui": None},
            ],
            "relationships": [],
            "decompose": True,
            "sub_queries": [
                {"query": "What is damage control surgery?", "intent": "conceptual"},
                {"query": "Steps for temporary abdominal closure", "intent": "procedural"},
            ],
            "retrieval_hint": "hybrid_search",
        },
    },
    "non_medical": {
        "query": "How does shock absorption work in car suspensions?",
        "output": {
            "medical": {"value": False, "confidence": 0.95},
            "intent": {"primary": None, "secondary": None},
            "entities": [],
            "relationships": [],
            "decompose": False,
            "sub_queries": None,
            "retrieval_hint": None,
        },
    },
}
