#!/usr/bin/env python3
"""Test hybrid prompt combining best strategies.

Results so far:
- V2 Few-Shot: 80% medical, 73% intent (best intent)
- Negative Examples: 87% medical, 64% intent (best medical)

Strategy: Combine negative examples WITH intent-specific few-shots.
Also try: Simpler binary medical check, then separate intent.
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

    # Non-medical - should be False
    ("How does shock absorption work in car suspensions?", False, None),
    ("What is the trauma of war on soldiers' families?", False, None),
    ("How do I perform a factory reset?", False, None),
    ("What is the capital of France?", False, None),
]


# Strategy 1: Hybrid - negative examples + intent-specific few-shots
PROMPT_HYBRID = """<|im_start|>system
Classify queries for a CLINICAL MEDICINE textbook. JSON only.

MEDICAL EXAMPLES:
"What is septic shock?" → {{"medical":true,"intent":"conceptual"}}
"How to perform thoracotomy?" → {{"medical":true,"intent":"procedural"}}
"How does hemorrhage affect BP?" → {{"medical":true,"intent":"relationship"}}
"Describe the Heimlich maneuver" → {{"medical":true,"intent":"procedural"}}
"Chapter 10 trauma" → {{"medical":true,"intent":"lookup"}}

NON-MEDICAL EXAMPLES:
"How do car shock absorbers work?" → {{"medical":false,"intent":null}}
"The emotional trauma of divorce" → {{"medical":false,"intent":null}}
"How to factory reset phone?" → {{"medical":false,"intent":null}}
"Psychological trauma in veterans" → {{"medical":false,"intent":null}}

INTENT RULES:
- "affect/effect/between/vs" → relationship
- "describe/technique/perform/steps" → procedural
- "chapter/section/find/reference" → lookup
- "what is/explain/define" → conceptual
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


# Strategy 2: Context-aware with domain hints
PROMPT_CONTEXT = """<|im_start|>system
You classify queries for a TRAUMA SURGERY medical textbook.

CONTEXT: This is about clinical medicine - surgeries, patient treatment, diseases.
NOT about: cars, phones, computers, emotions, psychology, geography.

"shock" = medical only if about patient blood pressure (NOT car suspension)
"trauma" = medical only if physical injury (NOT emotional/psychological)
"perform" = medical only if surgery/procedure (NOT device factory reset)

EXAMPLES:
{{"medical":true,"intent":"conceptual"}} ← "What is hemorrhagic shock?"
{{"medical":true,"intent":"procedural"}} ← "Describe thoracotomy technique"
{{"medical":true,"intent":"relationship"}} ← "How does X affect Y?"
{{"medical":true,"intent":"lookup"}} ← "Chapter 5 on bleeding"
{{"medical":false,"intent":null}} ← "Car shock absorbers", "emotional trauma", "factory reset"

Return JSON only.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


# Strategy 3: Ultra-minimal with key disambiguation
PROMPT_MINIMAL = """<|im_start|>system
Medical textbook query classifier. JSON only.

medical=true: clinical/surgery/patient care
medical=false: cars, phones, emotions, geography

Key terms:
- "shock" in patients=medical, in cars=not medical
- "trauma" physical injury=medical, emotional=not medical
- "describe/technique" surgery=procedural
- "affect/between" = relationship
- "chapter/section" = lookup
- "what is/explain" = conceptual
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""


# Strategy 4: Explicit disambiguation for problem terms
PROMPT_DISAMBIG = """<|im_start|>system
Classify for CLINICAL MEDICINE textbook. Return JSON.

DISAMBIGUATION (read carefully):
- "shock" → medical ONLY if about circulatory shock, blood pressure, patient
- "shock" → NOT medical if about cars, physics, electricity, surprise
- "trauma" → medical ONLY if about physical bodily injury
- "trauma" → NOT medical if about psychology, emotions, war effects on families
- "perform/reset" → NOT medical if about devices, computers, phones

FORMAT: {{"medical":true/false,"intent":"conceptual"|"procedural"|"relationship"|"lookup"|null}}

INTENT (only when medical=true):
- conceptual: "what is X", definitions
- procedural: "how to", "describe technique/maneuver", surgical steps
- relationship: "X affects Y", "between X and Y", comparisons
- lookup: chapter, section, page, find, reference
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
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
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
    print("MedQuery - Hybrid Prompt Strategies V3")
    print("=" * 70)

    model, tokenizer = load_model()

    # Warmup
    classify(model, tokenizer, "test", PROMPT_MINIMAL)

    results = []
    results.append(test_prompt(model, tokenizer, "Hybrid (Neg+Few-Shot)", PROMPT_HYBRID))
    results.append(test_prompt(model, tokenizer, "Context-Aware", PROMPT_CONTEXT))
    results.append(test_prompt(model, tokenizer, "Ultra-Minimal", PROMPT_MINIMAL))
    results.append(test_prompt(model, tokenizer, "Explicit Disambig", PROMPT_DISAMBIG))

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Medical':<10} {'Intent':<10} {'Time'}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} {r['med']:.0f}%{'':<5} {r['int']:.0f}%{'':<5} {r['time']:.0f}ms")

    best = max(results, key=lambda x: x['med'] + x['int'])
    print(f"\nBest: {best['name']} (Medical: {best['med']:.0f}%, Intent: {best['int']:.0f}%)")

    print("\n--- Previous Best Results ---")
    print("V2 Few-Shot:      Medical 80%, Intent 73%, 647ms")
    print("Negative Examples: Medical 87%, Intent 64%, 672ms")

    # Combined score comparison
    print("\n--- Combined Score (Medical + Intent) ---")
    prev_best = 80 + 73  # V2 Few-Shot
    for r in results:
        score = r['med'] + r['int']
        diff = score - prev_best
        indicator = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{r['name']:<25} {score:.0f} ({indicator}{abs(diff):.0f})")


if __name__ == "__main__":
    main()
