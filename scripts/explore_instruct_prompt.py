#!/usr/bin/env python3
"""Test improved instruction-based prompt design for Qwen2-1.5B MLX.

Issues from previous test:
1. JSON parsing failures (model output correct but parsing broke)
2. Intent classification confusion (relationship vs conceptual)
3. Lookup intent not recognized
4. False positives on non-medical queries

Solutions:
1. Fix JSON parsing to handle edge cases
2. Use few-shot examples in prompt
3. Clearer intent definitions with examples
4. Explicit instructions for edge cases
"""

import json
import re
import time
from mlx_lm import load, generate


# Test queries - same as complex queries
TEST_QUERIES = [
    # Multi-concept
    ("How does hemorrhagic shock affect coagulation cascade?", True, "relationship"),
    ("Explain trauma-induced coagulopathy and hypothermia connection", True, "relationship"),

    # Domain-specific
    ("What is the lethal triad in trauma?", True, "conceptual"),
    ("Explain FAST exam technique", True, "procedural"),
    ("How does permissive hypotension work?", True, "conceptual"),
    ("Describe the Mattox maneuver", True, "procedural"),

    # Lookup
    ("Chapter 15 damage control surgery", True, "lookup"),
    ("Find section on abdominal compartment syndrome", True, "lookup"),
    ("Reference for Glasgow Coma Scale", True, "lookup"),

    # Edge cases
    ("What is triage?", True, "conceptual"),
    ("What equipment is needed for surgery?", True, "procedural"),

    # Non-medical (should reject)
    ("How does shock absorption work in car suspensions?", False, None),
    ("What is the trauma of war on soldiers' families?", False, None),
    ("How do I perform a factory reset?", False, None),
    ("What is the capital of France?", False, None),
]


# Improved instruction-based prompt with few-shot examples
INSTRUCT_PROMPT_V1 = """<|im_start|>system
You are a medical query classifier for a trauma surgery textbook retrieval system.

TASK: Classify each query and return ONLY valid JSON.

OUTPUT FORMAT:
{{"medical": true/false, "intent": "conceptual"|"procedural"|"relationship"|"lookup"|null}}

INTENT DEFINITIONS:
- conceptual: "What is X?", definitions, explanations, pathophysiology
- procedural: "How to do X?", techniques, steps, surgical procedures
- relationship: "How does X affect Y?", comparisons, connections BETWEEN concepts
- lookup: References to chapters, sections, pages, or finding specific content

RULES:
1. "medical": true ONLY for clinical medicine, surgery, patient care
2. "medical": false for psychology, general science, non-clinical topics
3. If query asks about CONNECTION/EFFECT between two things → "relationship"
4. If query mentions chapter/section/page/find/reference → "lookup"
5. Return ONLY the JSON object, no explanations
<|im_end|>
<|im_start|>user
Query: {query}
<|im_end|>
<|im_start|>assistant
"""


# Even more structured with examples
INSTRUCT_PROMPT_V2 = """<|im_start|>system
Classify medical queries. Output JSON only.

EXAMPLES:
Query: What is sepsis?
{{"medical":true,"intent":"conceptual"}}

Query: How to perform thoracotomy?
{{"medical":true,"intent":"procedural"}}

Query: How does diabetes affect wound healing?
{{"medical":true,"intent":"relationship"}}

Query: Chapter 10 section on shock
{{"medical":true,"intent":"lookup"}}

Query: What is the weather today?
{{"medical":false,"intent":null}}

INTENT RULES:
- conceptual = definitions, "what is", explanations
- procedural = techniques, "how to", steps, procedures
- relationship = effects, comparisons, "affect", "between", "vs"
- lookup = chapter, section, page, find, reference

OUTPUT: JSON only, no text before or after.
<|im_end|>
<|im_start|>user
Query: {query}
<|im_end|>
<|im_start|>assistant
"""


# Minimal prompt for speed
INSTRUCT_PROMPT_V3 = """Classify this medical query. Return JSON only.

Rules:
- conceptual: what is, explain, define
- procedural: how to, steps, technique
- relationship: affect, compare, between, vs
- lookup: chapter, section, find, reference
- medical=false for non-clinical topics

Query: {query}
JSON:"""


def load_model():
    """Load model."""
    print("Loading Qwen2-1.5B-Instruct-4bit...")
    start = time.time()
    model, tokenizer = load("mlx-community/Qwen2-1.5B-Instruct-4bit")
    print(f"Loaded in {time.time() - start:.2f}s")
    return model, tokenizer


def robust_json_parse(text: str) -> dict:
    """Robust JSON parsing that handles various edge cases."""
    # Clean the text
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass

    # Find JSON object in text
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    # Try to fix common issues
    # Replace single quotes with double quotes
    fixed = text.replace("'", '"')
    try:
        return json.loads(fixed)
    except:
        pass

    # Handle true/false without quotes
    fixed = re.sub(r'\bTrue\b', 'true', text)
    fixed = re.sub(r'\bFalse\b', 'false', fixed)
    fixed = re.sub(r'\bNone\b', 'null', fixed)
    json_match = re.search(r'\{[^{}]*\}', fixed, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    return None


def classify_with_prompt(model, tokenizer, query: str, prompt_template: str, use_chat: bool = True) -> dict:
    """Classify using specified prompt template."""

    if use_chat and "<|im_start|>" in prompt_template:
        # Use raw prompt (already formatted)
        prompt = prompt_template.format(query=query)
    else:
        # Use chat template
        messages = [{"role": "user", "content": prompt_template.format(query=query)}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    start = time.perf_counter()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=80, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Parse response
    parsed = robust_json_parse(response)

    if parsed:
        return {
            "medical": parsed.get("medical"),
            "intent": parsed.get("intent"),
            "time_ms": elapsed_ms,
            "raw": response[:150],
            "parsed": True,
        }

    return {
        "medical": None,
        "intent": None,
        "time_ms": elapsed_ms,
        "raw": response[:150],
        "parsed": False,
    }


def test_prompt(model, tokenizer, name: str, prompt_template: str, use_chat: bool = True):
    """Test a prompt variant."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print("=" * 70)

    correct_medical = 0
    correct_intent = 0
    total = len(TEST_QUERIES)
    intent_total = 0
    times = []
    parse_failures = 0

    for query, expected_medical, expected_intent in TEST_QUERIES:
        result = classify_with_prompt(model, tokenizer, query, prompt_template, use_chat)
        times.append(result["time_ms"])

        if not result["parsed"]:
            parse_failures += 1

        # Check medical
        medical_match = result["medical"] == expected_medical
        if medical_match:
            correct_medical += 1

        # Check intent
        intent_match = None
        if expected_medical and expected_intent:
            intent_total += 1
            if result["intent"] == expected_intent:
                correct_intent += 1
                intent_match = True
            else:
                intent_match = False

        # Display
        med_status = "✓" if medical_match else "✗"
        int_status = ""
        if expected_intent:
            int_status = " ✓" if intent_match else " ✗"

        print(f"{result['time_ms']:5.0f}ms {med_status}{int_status} | {query[:50]}")

        if not medical_match or intent_match is False:
            print(f"        Expected: {expected_medical}/{expected_intent} | Got: {result['medical']}/{result['intent']}")

    # Summary
    med_acc = correct_medical / total * 100
    intent_acc = correct_intent / intent_total * 100 if intent_total > 0 else 0
    avg_time = sum(times) / len(times)

    print(f"\n--- {name} Results ---")
    print(f"Medical:  {correct_medical}/{total} ({med_acc:.0f}%)")
    print(f"Intent:   {correct_intent}/{intent_total} ({intent_acc:.0f}%)")
    print(f"Parse OK: {total - parse_failures}/{total}")
    print(f"Avg time: {avg_time:.0f}ms")

    return {
        "name": name,
        "medical_acc": med_acc,
        "intent_acc": intent_acc,
        "avg_time": avg_time,
        "parse_failures": parse_failures,
    }


def main():
    """Test different prompt designs."""
    print("=" * 70)
    print("MedQuery - Instruction Prompt Design Exploration")
    print("=" * 70)

    model, tokenizer = load_model()

    # Warmup
    print("\nWarmup...")
    classify_with_prompt(model, tokenizer, "test", INSTRUCT_PROMPT_V3, use_chat=False)

    results = []

    # Test each prompt variant
    results.append(test_prompt(model, tokenizer, "V1: Detailed Instructions", INSTRUCT_PROMPT_V1, use_chat=False))
    results.append(test_prompt(model, tokenizer, "V2: Few-Shot Examples", INSTRUCT_PROMPT_V2, use_chat=False))
    results.append(test_prompt(model, tokenizer, "V3: Minimal/Fast", INSTRUCT_PROMPT_V3, use_chat=False))

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Prompt':<30} {'Medical':<10} {'Intent':<10} {'Time':<10} {'Parse'}")
    print("-" * 70)

    for r in results:
        parse_str = f"{15 - r['parse_failures']}/15"
        print(f"{r['name']:<30} {r['medical_acc']:.0f}%{'':<5} {r['intent_acc']:.0f}%{'':<5} {r['avg_time']:.0f}ms{'':<4} {parse_str}")

    # Best result
    best = max(results, key=lambda x: x['medical_acc'] + x['intent_acc'])
    print(f"\nBest: {best['name']}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if best['medical_acc'] >= 90 and best['intent_acc'] >= 70:
        print("✓ Prompt engineering successful - ready for production")
    elif best['medical_acc'] >= 80:
        print("~ Acceptable - consider further tuning or larger model")
    else:
        print("✗ Need different approach (larger model or hybrid system)")


if __name__ == "__main__":
    main()
