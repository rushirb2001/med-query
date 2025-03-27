#!/usr/bin/env python3
"""Test alternative prompt strategies for better classification.

Issues from V2 (best so far - 80% medical, 73% intent):
1. False positives: "shock absorption in cars", "factory reset", "trauma of war"
2. Procedural confusion: "Describe X maneuver" → conceptual (should be procedural)
3. Relationship confusion: "X and Y connection" → conceptual (should be relationship)

New strategies:
1. Negative examples - explicitly show non-medical cases
2. Keyword triggers - very explicit mapping
3. Step-by-step reasoning - think first, classify second
4. Strict gatekeeper - default to non-medical unless clearly clinical
"""

import json
import re
import time
from mlx_lm import load, generate


TEST_QUERIES = [
    # Medical - should be True
    ("How does hemorrhagic shock affect coagulation cascade?", True, "relationship"),
    ("Explain trauma-induced coagulopathy and hypothermia connection", True, "relationship"),
    ("What is the lethal triad in trauma?", True, "conceptual"),
    ("Explain FAST exam technique", True, "procedural"),
    ("How does permissive hypotension work?", True, "conceptual"),
    ("Describe the Mattox maneuver", True, "procedural"),
    ("Chapter 15 damage control surgery", True, "lookup"),
    ("Find section on abdominal compartment syndrome", True, "lookup"),
    ("Reference for Glasgow Coma Scale", True, "lookup"),
    ("What is triage?", True, "conceptual"),
    ("What equipment is needed for surgery?", True, "procedural"),

    # Non-medical - should be False (these are the problem cases)
    ("How does shock absorption work in car suspensions?", False, None),
    ("What is the trauma of war on soldiers' families?", False, None),
    ("How do I perform a factory reset?", False, None),
    ("What is the capital of France?", False, None),
]


# Strategy 1: Negative examples with explicit non-medical cases
PROMPT_NEGATIVE = """<|im_start|>system
Classify queries for a CLINICAL MEDICINE textbook. Return JSON only.

MEDICAL (true):
- Patient care, surgery, diagnosis, treatment
- Clinical procedures, surgical techniques
- Pathophysiology, anatomy, pharmacology

NOT MEDICAL (false):
- Psychology, emotions, mental trauma (unless clinical psychiatry)
- Engineering, mechanics, physics
- Computers, technology, factory/device operations
- Geography, history, general knowledge

EXAMPLES:
"What is septic shock?" → {{"medical":true,"intent":"conceptual"}}
"How to perform appendectomy?" → {{"medical":true,"intent":"procedural"}}
"How does diabetes affect wounds?" → {{"medical":true,"intent":"relationship"}}
"Chapter 5 on trauma" → {{"medical":true,"intent":"lookup"}}
"How do car shock absorbers work?" → {{"medical":false,"intent":null}}
"The emotional trauma of divorce" → {{"medical":false,"intent":null}}
"How to factory reset my phone?" → {{"medical":false,"intent":null}}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


# Strategy 2: Keyword triggers with explicit mappings
PROMPT_KEYWORDS = """<|im_start|>system
Medical query classifier. JSON output only.

STEP 1 - Is it CLINICAL medicine?
YES if: patient, surgery, diagnosis, treatment, disease, procedure, anatomy, wound, blood, organ
NO if: car, phone, computer, device, emotional, psychology, geography, weather, cooking

STEP 2 - Intent (only if medical=true):
- PROCEDURAL if: "how to", "describe [procedure]", "technique", "maneuver", "steps", "perform"
- RELATIONSHIP if: "affect", "effect", "between", "connection", "vs", "compare"
- LOOKUP if: "chapter", "section", "page", "find", "reference", "where"
- CONCEPTUAL if: "what is", "explain", "define" (default for definitions)

Output: {{"medical":true/false,"intent":"procedural"|"relationship"|"lookup"|"conceptual"|null}}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


# Strategy 3: Strict gatekeeper - default to non-medical
PROMPT_STRICT = """<|im_start|>system
You are a STRICT medical content filter. When in doubt, classify as non-medical.

ONLY mark as medical if the query is CLEARLY about:
- Clinical patient care or treatment
- Surgical procedures or techniques
- Medical diagnosis or pathophysiology
- Anatomy, pharmacology, or clinical medicine

Mark as NON-MEDICAL if:
- Could be interpreted as non-clinical (psychology, emotions)
- Involves technology, devices, or engineering
- Is ambiguous or could apply to non-medical contexts

Format: {{"medical":true/false,"intent":"conceptual"|"procedural"|"relationship"|"lookup"|null}}

Intent rules (only if medical):
- "describe/technique/maneuver/perform" → procedural
- "affect/effect/between/compare" → relationship
- "chapter/section/find/reference" → lookup
- Otherwise → conceptual
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


# Strategy 4: Two-step reasoning
PROMPT_REASONING = """<|im_start|>system
Classify medical queries in two steps.

STEP 1: Is this about CLINICAL medicine (surgery, patient care, disease treatment)?
- "shock" in cars = NO (engineering)
- "shock" in patients = YES (medical)
- "trauma" emotional = NO (psychology)
- "trauma" physical injury = YES (medical)
- "reset" device = NO (technology)
- "perform" surgery = YES (medical)

STEP 2: If medical, what's the intent?
- Describes HOW to do something → procedural
- Asks about EFFECT/CONNECTION → relationship
- References CHAPTER/SECTION → lookup
- Asks WHAT something is → conceptual

Output JSON: {{"medical":true/false,"intent":"procedural"|"relationship"|"lookup"|"conceptual"|null}}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


def load_model():
    print("Loading Qwen2-1.5B-Instruct-4bit...")
    start = time.time()
    model, tokenizer = load("mlx-community/Qwen2-1.5B-Instruct-4bit")
    print(f"Loaded in {time.time() - start:.2f}s")
    return model, tokenizer


def robust_parse(text: str) -> dict:
    text = text.strip()
    # Find JSON
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            # Fix common issues
            fixed = match.group()
            fixed = re.sub(r'\bTrue\b', 'true', fixed)
            fixed = re.sub(r'\bFalse\b', 'false', fixed)
            fixed = re.sub(r'\bNone\b', 'null', fixed)
            try:
                return json.loads(fixed)
            except:
                pass
    return None


def classify(model, tokenizer, query: str, prompt_template: str) -> dict:
    prompt = prompt_template.format(query=query)

    start = time.perf_counter()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    parsed = robust_parse(response)
    if parsed:
        return {
            "medical": parsed.get("medical"),
            "intent": parsed.get("intent"),
            "time_ms": elapsed_ms,
            "raw": response[:100],
        }
    return {"medical": None, "intent": None, "time_ms": elapsed_ms, "raw": response[:100]}


def test_prompt(model, tokenizer, name: str, prompt_template: str):
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print("=" * 70)

    med_correct = 0
    int_correct = 0
    int_total = 0
    times = []

    for query, exp_med, exp_int in TEST_QUERIES:
        result = classify(model, tokenizer, query, prompt_template)
        times.append(result["time_ms"])

        med_ok = result["medical"] == exp_med
        if med_ok:
            med_correct += 1

        int_ok = None
        if exp_med and exp_int:
            int_total += 1
            int_ok = result["intent"] == exp_int
            if int_ok:
                int_correct += 1

        # Display
        status = "✓" if med_ok else "✗"
        int_status = ""
        if exp_int:
            int_status = " ✓" if int_ok else " ✗"

        print(f"{result['time_ms']:5.0f}ms {status}{int_status} | {query[:45]}")

        if not med_ok or int_ok is False:
            print(f"        Exp: {exp_med}/{exp_int} | Got: {result['medical']}/{result['intent']}")

    med_acc = med_correct / len(TEST_QUERIES) * 100
    int_acc = int_correct / int_total * 100 if int_total > 0 else 0

    print(f"\n--- Results ---")
    print(f"Medical: {med_correct}/{len(TEST_QUERIES)} ({med_acc:.0f}%)")
    print(f"Intent:  {int_correct}/{int_total} ({int_acc:.0f}%)")
    print(f"Avg time: {sum(times)/len(times):.0f}ms")

    return {"name": name, "med": med_acc, "int": int_acc, "time": sum(times)/len(times)}


def main():
    print("=" * 70)
    print("MedQuery - Alternative Prompt Strategies")
    print("=" * 70)

    model, tokenizer = load_model()

    # Warmup
    classify(model, tokenizer, "test", PROMPT_KEYWORDS)

    results = []
    results.append(test_prompt(model, tokenizer, "Negative Examples", PROMPT_NEGATIVE))
    results.append(test_prompt(model, tokenizer, "Keyword Triggers", PROMPT_KEYWORDS))
    results.append(test_prompt(model, tokenizer, "Strict Gatekeeper", PROMPT_STRICT))
    results.append(test_prompt(model, tokenizer, "Two-Step Reasoning", PROMPT_REASONING))

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Medical':<10} {'Intent':<10} {'Time'}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} {r['med']:.0f}%{'':<5} {r['int']:.0f}%{'':<5} {r['time']:.0f}ms")

    best = max(results, key=lambda x: x['med'] + x['int'])
    print(f"\nBest: {best['name']} (Medical: {best['med']:.0f}%, Intent: {best['int']:.0f}%)")

    # Compare to previous best
    print("\nPrevious best (V2 Few-Shot): Medical 80%, Intent 73%, 647ms")


if __name__ == "__main__":
    main()
