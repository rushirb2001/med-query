"""Modular prompt components for query classification.

Provides structured prompts for:
- Domain boundary detection (medical vs non-medical)
- Intent classification (conceptual, procedural, relationship, lookup)
- Entity extraction
- Relationship extraction
- Query decomposition
"""

from typing import Literal

from ..logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# OUTPUT SCHEMA STRING
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
# PROMPT SECTIONS
# =============================================================================

class PromptSections:
    """Modular prompt sections for medical query classification."""

    DOMAIN_BOUNDARY = """## MEDICAL DOMAIN CLASSIFICATION

Classify queries for a CLINICAL MEDICINE (trauma surgery) textbook.

MEDICAL (value=true):
- Clinical patient care, diagnosis, treatment
- Surgical procedures and techniques
- Pathophysiology, anatomy, pharmacology
- Medical conditions, diseases, injuries

NOT MEDICAL (value=false):
- Psychology, emotions, mental trauma (unless clinical psychiatry)
- Engineering, mechanics, physics, computers
- Geography, history, business, finance

DISAMBIGUATION:
| Term | Medical | Non-Medical |
|------|---------|-------------|
| shock | circulatory/patient shock | car suspension, surprise |
| trauma | physical injury | emotional, psychological |
| procedure | medical intervention | business process |"""

    INTENT_CLASSIFICATION = """## INTENT CLASSIFICATION

| Intent | Pattern | Example |
|--------|---------|---------|
| conceptual | what is, explain, define | "What is hemorrhagic shock?" |
| procedural | how to, steps, technique | "How to perform thoracotomy?" |
| relationship | affect, compare, between, vs | "How does X affect Y?" |
| lookup | chapter, section, find | "Chapter 10 on trauma" |

DECISION RULES:
1. "How does X affect Y?" -> relationship (NOT procedural)
2. "Describe [procedure name]" -> procedural
3. "Describe [condition]" -> conceptual
4. Chapter/section/page mentioned -> lookup"""

    ENTITY_EXTRACTION = """## ENTITY EXTRACTION

| Type | Description | Examples |
|------|-------------|----------|
| condition | Diseases, injuries | sepsis, TBI, hemorrhagic shock |
| procedure | Medical procedures | thoracotomy, intubation |
| anatomy | Body parts, organs | liver, femoral artery |
| process | Physiological processes | coagulation, hemostasis |
| concept | Abstract concepts | triage, damage control |
| medication | Drugs, treatments | epinephrine, TXA |"""

    RELATIONSHIP_EXTRACTION = """## RELATIONSHIP EXTRACTION

| Type | Pattern | Example |
|------|---------|---------|
| affects | X influences Y | "shock affects coagulation" |
| causes | X leads to Y | "trauma causes coagulopathy" |
| treats | X treats Y | "TXA treats hemorrhage" |
| indicates | X is sign of Y | "tachycardia indicates shock" |
| compared_to | X vs Y | "open vs laparoscopic" |"""

    DECOMPOSITION = """## QUERY DECOMPOSITION

Decompose when query:
- Contains "and" joining different question types
- Asks multiple distinct questions
- Combines conceptual + procedural intents

Example: "What is damage control surgery and what are the steps?"
-> [{"query": "What is damage control surgery?", "intent": "conceptual"},
    {"query": "What are the steps?", "intent": "procedural"}]"""

    OUTPUT_FORMAT = f"""## OUTPUT FORMAT

Return ONLY valid JSON:
{OUTPUT_SCHEMA_STR}

IMPORTANT: No explanations, no markdown, ONLY the JSON object."""


# =============================================================================
# PROMPT PRESETS
# =============================================================================

PROMPT_PRESETS: dict[str, str] = {
    "minimal": """You are a medical query classifier. Analyze queries and output JSON with:
- is_medical: boolean (true if medical domain)
- confidence: float 0.0-1.0
- primary_intent: "conceptual" | "procedural" | "relationship" | "lookup" | null
- entities: array of {text, type, cui} where type is condition|procedure|anatomy|process|concept|medication
- relationships: array of {source, target, type} where type is affects|causes|treats|indicates|compared_to

Output ONLY valid JSON, no explanation.""",

    "standard": f"""You are a medical query classifier for a trauma surgery textbook.

{PromptSections.DOMAIN_BOUNDARY}

{PromptSections.INTENT_CLASSIFICATION}

{PromptSections.OUTPUT_FORMAT}""",

    "comprehensive": f"""You are a medical query classifier for a trauma surgery textbook retrieval system.
Analyze the query and return structured JSON output.

{PromptSections.DOMAIN_BOUNDARY}

{PromptSections.ENTITY_EXTRACTION}

{PromptSections.INTENT_CLASSIFICATION}

{PromptSections.RELATIONSHIP_EXTRACTION}

{PromptSections.DECOMPOSITION}

{PromptSections.OUTPUT_FORMAT}""",
}


# =============================================================================
# PROMPT FACTORY
# =============================================================================

PromptPreset = Literal["minimal", "standard", "comprehensive"]


def get_prompt(preset: PromptPreset = "standard") -> str:
    """Get a prompt by preset name.

    Args:
        preset: One of "minimal", "standard", "comprehensive"

    Returns:
        System prompt string
    """
    if preset not in PROMPT_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PROMPT_PRESETS.keys())}")
    logger.debug(f"Using prompt preset: {preset} ({len(PROMPT_PRESETS[preset])} chars)")
    return PROMPT_PRESETS[preset]


def get_section_prompt(*sections: str) -> str:
    """Build a custom prompt from specific sections.

    Args:
        *sections: Section names to include (domain, intent, entity, relationship, decomposition)

    Returns:
        Combined prompt string
    """
    section_map = {
        "domain": PromptSections.DOMAIN_BOUNDARY,
        "intent": PromptSections.INTENT_CLASSIFICATION,
        "entity": PromptSections.ENTITY_EXTRACTION,
        "relationship": PromptSections.RELATIONSHIP_EXTRACTION,
        "decomposition": PromptSections.DECOMPOSITION,
        "output": PromptSections.OUTPUT_FORMAT,
    }

    parts = ["You are a medical query classifier."]
    for section in sections:
        if section in section_map:
            parts.append(section_map[section])

    # Always include output format
    if "output" not in sections:
        parts.append(PromptSections.OUTPUT_FORMAT)

    return "\n\n".join(parts)


# =============================================================================
# ADAPTIVE PROMPT SELECTOR
# =============================================================================

class PromptSelector:
    """Intelligently selects prompts based on query characteristics."""

    # Pattern definitions for query analysis
    RELATIONSHIP_PATTERNS = frozenset([
        "affect", "effect", "between", "compare", "vs", "versus",
        "relationship", "connection", "influence", "impact"
    ])

    LOOKUP_PATTERNS = frozenset([
        "chapter", "section", "page", "find", "where is", "reference",
        "locate", "which part"
    ])

    PROCEDURAL_PATTERNS = frozenset([
        "how to", "steps for", "technique", "perform", "procedure for",
        "method for", "process of"
    ])

    COMPLEX_INDICATORS = frozenset([" and what", " and how", "? and"])

    def __init__(self, default_preset: PromptPreset = "standard"):
        """Initialize with a default preset.

        Args:
            default_preset: Fallback preset when no specific pattern matches
        """
        self.default_preset = default_preset
        self._cache: dict[str, str] = {}

    def select(self, query: str) -> str:
        """Select the most appropriate prompt for a query.

        Args:
            query: The user query to analyze

        Returns:
            Selected system prompt string
        """
        query_lower = query.lower()

        # Check for relationship patterns -> comprehensive
        if self._has_pattern(query_lower, self.RELATIONSHIP_PATTERNS):
            logger.debug(f"Selected 'comprehensive' for relationship query: {query[:50]}...")
            return get_prompt("comprehensive")

        # Check for complex queries with multiple intents -> comprehensive
        if self._is_complex_query(query_lower):
            logger.debug(f"Selected 'comprehensive' for complex query: {query[:50]}...")
            return get_prompt("comprehensive")

        # Check for simple lookup -> minimal
        if self._has_pattern(query_lower, self.LOOKUP_PATTERNS):
            logger.debug(f"Selected 'minimal' for lookup query: {query[:50]}...")
            return get_prompt("minimal")

        # Default to configured preset
        logger.debug(f"Using default preset '{self.default_preset}' for: {query[:50]}...")
        return get_prompt(self.default_preset)

    def _has_pattern(self, text: str, patterns: frozenset[str]) -> bool:
        """Check if text contains any of the patterns."""
        return any(p in text for p in patterns)

    def _is_complex_query(self, text: str) -> bool:
        """Detect complex queries that need decomposition."""
        # Multiple question marks
        if text.count("?") > 1:
            return True

        # "and what/how" patterns
        if self._has_pattern(text, self.COMPLEX_INDICATORS):
            return True

        return False

    def get_sections_for_query(self, query: str) -> list[str]:
        """Determine which prompt sections are needed for a query.

        Args:
            query: The user query

        Returns:
            List of section names needed
        """
        query_lower = query.lower()
        sections = ["domain", "intent"]  # Always needed

        if self._has_pattern(query_lower, self.RELATIONSHIP_PATTERNS):
            sections.extend(["entity", "relationship"])

        if self._is_complex_query(query_lower):
            sections.append("decomposition")

        return sections
