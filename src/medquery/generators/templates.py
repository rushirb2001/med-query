"""Query templates for each category and complexity level.

Templates provide context and examples for Claude to generate
diverse, realistic queries with proper annotations.
"""

# =============================================================================
# MEDICAL CONCEPTUAL TEMPLATES
# =============================================================================

TEMPLATE_MEDICAL_CONCEPTUAL = """You are generating medical conceptual queries for a trauma surgery textbook retrieval system.

Conceptual queries ask "What is X?", "Explain Y", "Define Z" - seeking definitions and explanations.

Medical domains to cover:
- Trauma & Shock (hemorrhagic, septic, cardiogenic, neurogenic)
- Surgical Conditions (appendicitis, cholecystitis, bowel obstruction, hernias)
- Vascular (aneurysms, dissections, DVT, PE, compartment syndrome)
- Respiratory (pneumothorax, hemothorax, ARDS, pulmonary contusion)
- Neurological (TBI, epidural/subdural hematoma, herniation)
- Cardiac (tamponade, myocardial contusion, aortic rupture)
- Burns & Wounds (degrees, rule of nines, wound healing)
- Infection & Sepsis (SIRS, necrotizing fasciitis, septic shock)
- Anatomy & Physiology (coagulation cascade, hemostasis, cardiac output)

Entity types to annotate:
- condition: diseases, syndromes, injuries
- anatomy: body parts, organs, structures
- process: physiological processes
- concept: abstract medical concepts

Examples:
- "What is hemorrhagic shock?" -> entities: [{"text": "hemorrhagic shock", "type": "condition"}]
- "Explain the coagulation cascade" -> entities: [{"text": "coagulation cascade", "type": "process"}]
- "Define compartment syndrome" -> entities: [{"text": "compartment syndrome", "type": "condition"}]
"""


# =============================================================================
# MEDICAL PROCEDURAL TEMPLATES
# =============================================================================

TEMPLATE_MEDICAL_PROCEDURAL = """You are generating medical procedural queries for a trauma surgery textbook retrieval system.

Procedural queries ask "How to do X?", "Steps for Y", "Technique for Z" - seeking step-by-step instructions.

Procedure categories to cover:
- Surgical Procedures (thoracotomy, laparotomy, splenectomy, appendectomy)
- Trauma Procedures (FAST exam, damage control surgery, REBOA, Pringle maneuver)
- Airway & Breathing (intubation, cricothyrotomy, chest tube, needle decompression)
- Vascular Access (central line, arterial line, intraosseous access)
- Vascular Surgery (fasciotomy, embolectomy, bypass grafting)
- Neuro Procedures (craniotomy, burr hole, ICP monitor, lumbar puncture)
- Cardiac Procedures (pericardiocentesis, cardiac massage, CPR)
- Wound Care (suturing, debridement, skin grafting, NPWT)
- Orthopedic (fracture reduction, splinting, external fixation)

Entity types to annotate:
- procedure: surgical/medical procedures
- anatomy: body parts involved
- condition: conditions being treated

Examples:
- "How to perform a thoracotomy?" -> entities: [{"text": "thoracotomy", "type": "procedure"}]
- "Describe the Mattox maneuver" -> entities: [{"text": "Mattox maneuver", "type": "procedure"}]
- "Steps for chest tube insertion" -> entities: [{"text": "chest tube insertion", "type": "procedure"}]
"""


# =============================================================================
# MEDICAL RELATIONSHIP TEMPLATES
# =============================================================================

TEMPLATE_MEDICAL_RELATIONSHIP = """You are generating medical relationship queries for a trauma surgery textbook retrieval system.

Relationship queries ask "How does X affect Y?", "Compare X vs Y", "Connection between X and Y".

Relationship types to cover:
- Pathophysiology: how conditions affect physiological processes
- Comparative: comparing procedures, treatments, conditions
- Treatment effects: how interventions affect outcomes
- Causal: what causes conditions or complications

Include relationship annotations:
- affects: X changes/impacts Y
- causes: X leads to Y
- treats: X is used to treat Y
- indicates: X suggests/points to Y
- compared_to: X is being compared with Y

Examples:
- "How does hemorrhagic shock affect coagulation?" ->
  relationships: [{"source": "hemorrhagic shock", "target": "coagulation", "type": "affects"}]
- "Compare open vs laparoscopic surgery" ->
  relationships: [{"source": "open surgery", "target": "laparoscopic surgery", "type": "compared_to"}]
- "Connection between trauma and coagulopathy" ->
  relationships: [{"source": "trauma", "target": "coagulopathy", "type": "causes"}]
"""


# =============================================================================
# MEDICAL LOOKUP TEMPLATES
# =============================================================================

TEMPLATE_MEDICAL_LOOKUP = """You are generating medical lookup queries for a trauma surgery textbook retrieval system.

Lookup queries reference specific locations in the textbook: chapters, sections, pages, tables, figures.

Lookup patterns:
- "Chapter X on Y"
- "Find section about Z"
- "Where is X discussed?"
- "Reference for Y"
- "Table/Figure for Z"

Topics to reference:
- Trauma management (shock, hemorrhage, TBI, burns)
- Surgical procedures (damage control, emergency surgery)
- Assessment tools (GCS, trauma scores, ATLS)
- Algorithms (resuscitation, triage, treatment protocols)
- Anatomy diagrams and reference tables

Examples:
- "Chapter 15 damage control surgery"
- "Find section on abdominal compartment syndrome"
- "Reference for Glasgow Coma Scale"
- "Where is massive transfusion protocol discussed?"
"""


# =============================================================================
# MEDICAL COMPLEX TEMPLATES (L3-L5)
# =============================================================================

TEMPLATE_MEDICAL_COMPLEX = """You are generating complex medical queries that require decomposition for a trauma surgery textbook.

Complex queries have multiple parts, require multiple retrieval strategies, and may need decomposition.

Complexity patterns:
- L3: Multi-concept within same domain ("How does X affect Y and what are the Z?")
- L4: Requires decomposition ("Explain A, then describe how to do B")
- L5: Multi-intent with entity+relationship extraction

Include decomposition for L4+ queries:
- Break into 2-3 sub-queries
- Each sub-query has its own intent

Include retrieval_strategy hints:
- vector_search: semantic similarity
- graph_traversal: knowledge graph paths
- hybrid_search: combined approach

Examples:
L3: "How does hemorrhagic shock affect the coagulation cascade and what are the clinical signs?"
L4: "What is damage control surgery and what are the steps for temporary abdominal closure?"
L5: "Compare REBOA vs thoracotomy for hemorrhage control, describing indications and technique for each"

For L4+ queries, provide decomposition:
[
  {"query": "What is damage control surgery?", "intent": "conceptual"},
  {"query": "Steps for temporary abdominal closure", "intent": "procedural"}
]
"""


# =============================================================================
# NON-MEDICAL TEMPLATES
# =============================================================================

TEMPLATE_NON_MEDICAL = """You are generating NON-MEDICAL queries to test false positive rejection.

These queries should NOT be classified as medical. They use similar words but in non-medical contexts.

Categories of non-medical queries:
1. Technology: factory reset, software debugging, device configuration
2. Automotive/Engineering: car shock absorbers, suspension systems, mechanics
3. Psychology/Emotions: emotional trauma, psychological shock, grief, stress
4. Geography/General: capitals, mountains, weather, history
5. Daily Life: cooking, cleaning, home repair
6. Business/Finance: market crash, economic shock, audits
7. Safety/Emergency: fire evacuation, earthquake drills (non-medical emergency)

Key disambiguation:
- "shock" in cars = NOT medical
- "trauma" emotional = NOT medical
- "perform" reset = NOT medical
- "procedure" business = NOT medical

Examples:
- "How does shock absorption work in car suspensions?" (engineering)
- "What is the trauma of war on soldiers' families?" (psychology)
- "How do I perform a factory reset?" (technology)
- "What is the procedure for filing taxes?" (business)
"""


# =============================================================================
# EDGE CASE TEMPLATES
# =============================================================================

TEMPLATE_EDGE_CASES = """You are generating edge case queries to test disambiguation accuracy.

Edge cases are queries that could be interpreted either way but have a correct classification.

Categories:
1. Medical despite simple phrasing:
   - "What is shock?" (medical - circulatory shock)
   - "Define trauma" (medical - physical injury)

2. Medical abbreviations:
   - "What is BP?" (blood pressure)
   - "Explain GCS" (Glasgow Coma Scale)

3. Medical context clues:
   - "Patient in shock treatment"
   - "Trauma patient assessment"

4. Non-medical despite medical-sounding words:
   - "Shock value in comedy"
   - "Electric shock hazard"
   - "Shell shock history" (historical term, not clinical)
   - "Trauma in literature"

5. Ambiguous but medical:
   - "Emergency room procedures" (medical context)
   - "Operating room setup" (medical context)

6. Ambiguous but non-medical:
   - "Shock therapy for the economy" (metaphor)
   - "Surgical precision in business" (metaphor)

Include confidence scores:
- High confidence (0.9+): clearly medical or clearly non-medical
- Medium confidence (0.7-0.9): somewhat ambiguous
- Low confidence (0.5-0.7): very ambiguous
"""


# =============================================================================
# ENTITY EXTRACTION TEMPLATES
# =============================================================================

TEMPLATE_ENTITY_EXTRACTION = """You are generating queries specifically for entity extraction testing.

Each query should contain multiple extractable medical entities with type annotations.

Entity types:
- condition: diseases, syndromes, injuries (sepsis, TBI, fracture)
- procedure: surgical/medical procedures (thoracotomy, intubation)
- anatomy: body parts, organs, structures (liver, femoral artery)
- process: physiological processes (coagulation, inflammation)
- concept: abstract medical concepts (triage, hemostasis)
- medication: drugs, treatments (epinephrine, TXA, blood products)

Generate queries with 2-5 entities each.

Examples:
- "What is the relationship between sepsis and acute kidney injury?"
  entities: [
    {"text": "sepsis", "type": "condition", "cui": "C0243026"},
    {"text": "acute kidney injury", "type": "condition", "cui": "C2609414"}
  ]

- "How does epinephrine affect cardiac output in hemorrhagic shock?"
  entities: [
    {"text": "epinephrine", "type": "medication", "cui": "C0014563"},
    {"text": "cardiac output", "type": "process", "cui": "C0007165"},
    {"text": "hemorrhagic shock", "type": "condition", "cui": "C0038454"}
  ]
"""


# =============================================================================
# RELATIONSHIP EXTRACTION TEMPLATES
# =============================================================================

TEMPLATE_RELATIONSHIP_EXTRACTION = """You are generating queries specifically for relationship extraction testing.

Each query should contain extractable relationships between medical concepts.

Relationship types:
- affects: X influences/changes Y
- causes: X leads to/produces Y
- treats: X is used to treat Y
- indicates: X is a sign/symptom of Y
- has_property: X has characteristic Y
- compared_to: X is being compared with Y
- part_of: X is a component of Y
- precedes: X comes before Y (in time or procedure)

Generate queries with 1-3 relationships each.

Examples:
- "How does massive transfusion cause dilutional coagulopathy?"
  relationships: [
    {"source": "massive transfusion", "target": "dilutional coagulopathy", "type": "causes"}
  ]

- "Compare the effectiveness of TXA vs fresh frozen plasma for trauma hemorrhage"
  relationships: [
    {"source": "TXA", "target": "fresh frozen plasma", "type": "compared_to"},
    {"source": "TXA", "target": "trauma hemorrhage", "type": "treats"},
    {"source": "fresh frozen plasma", "target": "trauma hemorrhage", "type": "treats"}
  ]
"""


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

QUERY_TEMPLATES = {
    # Medical categories
    "medical_conceptual": TEMPLATE_MEDICAL_CONCEPTUAL,
    "medical_procedural": TEMPLATE_MEDICAL_PROCEDURAL,
    "medical_relationship": TEMPLATE_MEDICAL_RELATIONSHIP,
    "medical_lookup": TEMPLATE_MEDICAL_LOOKUP,
    "medical_complex": TEMPLATE_MEDICAL_COMPLEX,

    # Non-medical categories
    "non_medical": TEMPLATE_NON_MEDICAL,

    # Special categories
    "edge_cases": TEMPLATE_EDGE_CASES,
    "entity_extraction": TEMPLATE_ENTITY_EXTRACTION,
    "relationship_extraction": TEMPLATE_RELATIONSHIP_EXTRACTION,
}
