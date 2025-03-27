"""Master prompt combining all classification components.

Provides the unified prompt for Qwen2-1.5B MLX with:
- Domain boundary detection
- Entity extraction
- Intent classification
- Relationship extraction
- Query decomposition
"""

from .schemas import OUTPUT_SCHEMA_STR, EXAMPLE_OUTPUTS


class MasterPrompt:
    """Unified prompt for medical query classification."""

    # =========================================================================
    # SECTION 1: DOMAIN BOUNDARY
    # =========================================================================
    DOMAIN_BOUNDARY = """## SECTION 1: MEDICAL DOMAIN CLASSIFICATION

You are classifying queries for a CLINICAL MEDICINE (trauma surgery) textbook.

### MEDICAL (value=true):
- Clinical patient care, diagnosis, treatment
- Surgical procedures and techniques
- Pathophysiology, anatomy, pharmacology
- Medical conditions, diseases, injuries
- Clinical assessment tools (GCS, trauma scores)

### NOT MEDICAL (value=false):
- Psychology, emotions, mental trauma (unless clinical psychiatry)
- Engineering, mechanics, physics
- Computers, technology, device operations
- Geography, history, general knowledge
- Business, finance, economics
- Metaphorical use of medical terms

### DISAMBIGUATION RULES:
| Term | Medical Context | Non-Medical Context |
|------|-----------------|---------------------|
| shock | circulatory/patient shock | car suspension, surprise, electric |
| trauma | physical injury | emotional, psychological, literary |
| perform | surgery, procedure | device reset, business task |
| procedure | medical intervention | business process, legal |
| reset | (not medical) | factory reset, device reset |"""

    # =========================================================================
    # SECTION 2: ENTITY EXTRACTION
    # =========================================================================
    ENTITY_EXTRACTION = """## SECTION 2: ENTITY EXTRACTION

Extract medical entities with type classification:

| Type | Description | Examples |
|------|-------------|----------|
| condition | Diseases, syndromes, injuries | sepsis, TBI, hemorrhagic shock |
| procedure | Surgical/medical procedures | thoracotomy, intubation, FAST exam |
| anatomy | Body parts, organs, structures | liver, femoral artery, peritoneum |
| process | Physiological processes | coagulation, hemostasis, inflammation |
| concept | Abstract medical concepts | triage, damage control, golden hour |
| medication | Drugs and treatments | epinephrine, TXA, blood products |

### ABBREVIATION EXPANSION:
- BP = blood pressure (process)
- HR = heart rate (process)
- GCS = Glasgow Coma Scale (concept)
- TBI = traumatic brain injury (condition)
- ARDS = acute respiratory distress syndrome (condition)
- DVT = deep vein thrombosis (condition)
- PE = pulmonary embolism (condition)"""

    # =========================================================================
    # SECTION 3: INTENT CLASSIFICATION
    # =========================================================================
    INTENT_CLASSIFICATION = """## SECTION 3: INTENT CLASSIFICATION

Classify the primary intent (and secondary if applicable):

| Intent | Pattern | Example |
|--------|---------|---------|
| conceptual | what is, explain, define | "What is hemorrhagic shock?" |
| procedural | how to, steps, technique, describe procedure | "How to perform thoracotomy?" |
| relationship | affect, compare, between, vs, connection | "How does X affect Y?" |
| lookup | chapter, section, find, reference, where | "Chapter 10 on trauma" |

### DECISION RULES:
1. If query asks "How does X affect Y?" → relationship (NOT procedural)
2. If query asks "Describe [procedure name]" → procedural
3. If query asks "Describe [condition]" → conceptual
4. If query mentions chapter/section/page → lookup
5. Default for definitions → conceptual"""

    # =========================================================================
    # SECTION 4: RELATIONSHIP EXTRACTION
    # =========================================================================
    RELATIONSHIP_EXTRACTION = """## SECTION 4: RELATIONSHIP EXTRACTION

Extract relationships between entities:

| Type | Pattern | Example |
|------|---------|---------|
| affects | X influences/changes Y | "shock affects coagulation" |
| causes | X leads to Y | "trauma causes coagulopathy" |
| treats | X treats condition Y | "TXA treats hemorrhage" |
| indicates | X is sign of Y | "tachycardia indicates shock" |
| compared_to | X vs Y | "open vs laparoscopic" |

### EXTRACTION FORMAT:
{"source": "entity1", "target": "entity2", "type": "relationship_type"}"""

    # =========================================================================
    # SECTION 5: DECOMPOSITION
    # =========================================================================
    DECOMPOSITION = """## SECTION 5: QUERY DECOMPOSITION

Complex queries should be decomposed when they:
- Contain "and" joining different question types
- Ask multiple distinct questions
- Combine conceptual + procedural intents

### DECOMPOSITION RULES:
1. Split on "and what/how" patterns
2. Split on question marks within query
3. Each sub-query gets its own intent

### EXAMPLE:
Query: "What is damage control surgery and what are the steps?"
Decomposition:
- {"query": "What is damage control surgery?", "intent": "conceptual"}
- {"query": "What are the steps for damage control surgery?", "intent": "procedural"}"""

    # =========================================================================
    # SECTION 6: OUTPUT FORMAT
    # =========================================================================
    OUTPUT_FORMAT = f"""## SECTION 6: OUTPUT FORMAT

Return ONLY valid JSON matching this schema:
{OUTPUT_SCHEMA_STR}

### EXAMPLES:

**Simple Conceptual:**
Query: "What is hemorrhagic shock?"
{{"medical":{{"value":true,"confidence":0.98}},"intent":{{"primary":"conceptual","secondary":null}},"entities":[{{"text":"hemorrhagic shock","type":"condition"}}],"relationships":[],"decompose":false,"sub_queries":null,"retrieval_hint":"vector_search"}}

**Relationship:**
Query: "How does trauma affect coagulation?"
{{"medical":{{"value":true,"confidence":0.97}},"intent":{{"primary":"relationship","secondary":null}},"entities":[{{"text":"trauma","type":"condition"}},{{"text":"coagulation","type":"process"}}],"relationships":[{{"source":"trauma","target":"coagulation","type":"affects"}}],"decompose":false,"sub_queries":null,"retrieval_hint":"graph_then_vector"}}

**Non-Medical:**
Query: "How do car shock absorbers work?"
{{"medical":{{"value":false,"confidence":0.95}},"intent":{{"primary":null,"secondary":null}},"entities":[],"relationships":[],"decompose":false,"sub_queries":null,"retrieval_hint":null}}"""

    @classmethod
    def get_full_prompt(cls) -> str:
        """Get the complete prompt with all sections."""
        return f"""<|im_start|>system
You are a medical query classifier for a trauma surgery textbook retrieval system.
Analyze the query and return structured JSON output.

{cls.DOMAIN_BOUNDARY}

{cls.ENTITY_EXTRACTION}

{cls.INTENT_CLASSIFICATION}

{cls.RELATIONSHIP_EXTRACTION}

{cls.DECOMPOSITION}

{cls.OUTPUT_FORMAT}

IMPORTANT: Return ONLY the JSON object, no explanations or markdown.
<|im_end|>
<|im_start|>user
{{query}}
<|im_end|>
<|im_start|>assistant
"""

    @classmethod
    def get_minimal_prompt(cls) -> str:
        """Get a minimal prompt for faster inference."""
        return """<|im_start|>system
Medical query classifier. Return JSON only.

MEDICAL: clinical medicine, surgery, patient care, diseases, procedures
NOT MEDICAL: cars, phones, emotions, psychology, geography, business

DISAMBIGUATION:
- "shock" in patients = medical; in cars = not medical
- "trauma" physical = medical; emotional = not medical
- "perform" surgery = medical; reset = not medical

INTENT (if medical):
- conceptual: what is, explain, define
- procedural: how to, describe technique, steps
- relationship: affect, compare, between, vs
- lookup: chapter, section, find

OUTPUT: {"medical":{"value":true/false,"confidence":0.0-1.0},"intent":{"primary":"...","secondary":null},"entities":[...],"relationships":[...],"decompose":false,"sub_queries":null,"retrieval_hint":"..."}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""

    @classmethod
    def get_prompt_for_intent(cls, intent: str) -> str:
        """Get a focused prompt for a specific intent type."""
        intent_prompts = {
            "conceptual": cls.DOMAIN_BOUNDARY + "\n" + cls.ENTITY_EXTRACTION,
            "procedural": cls.DOMAIN_BOUNDARY + "\n" + cls.ENTITY_EXTRACTION,
            "relationship": cls.DOMAIN_BOUNDARY + "\n" + cls.ENTITY_EXTRACTION + "\n" + cls.RELATIONSHIP_EXTRACTION,
            "lookup": cls.DOMAIN_BOUNDARY,
        }
        return intent_prompts.get(intent, cls.get_full_prompt())


# Convenience function for getting the prompt
def get_master_prompt(minimal: bool = False) -> str:
    """Get the master prompt.

    Args:
        minimal: If True, return minimal prompt for faster inference

    Returns:
        Prompt string with {query} placeholder
    """
    if minimal:
        return MasterPrompt.get_minimal_prompt()
    return MasterPrompt.get_full_prompt()
