#!/usr/bin/env python3
"""Test best performing model (Qwen2-1.5B MLX) on complex domain-specific queries.

This script validates classification on:
1. Complex multi-part medical queries
2. Edge cases and ambiguous queries
3. Real-world medical textbook retrieval patterns
4. Queries requiring decomposition
5. Domain-specific terminology (trauma surgery focus)

Best model from exploration: Qwen2-1.5B-Instruct-4bit (MLX)
- 100% accuracy on basic tests
- ~400ms per query
"""

import json
import time
from mlx_lm import load, generate


# Complex test queries organized by category
COMPLEX_QUERIES = {
    "multi_concept": [
        # Queries involving multiple medical concepts
        ("How does hemorrhagic shock affect coagulation cascade and what are the resuscitation endpoints?",
         True, "relationship"),
        ("Explain the pathophysiology of trauma-induced coagulopathy and its relationship to hypothermia",
         True, "conceptual"),
        ("What is the connection between massive transfusion protocol and dilutional coagulopathy?",
         True, "relationship"),
    ],

    "multi_intent": [
        # Queries that could map to multiple intents
        ("What is damage control surgery and how is it performed in abdominal trauma?",
         True, "conceptual"),  # Could be procedural too
        ("Explain the steps of emergency thoracotomy and when it's indicated",
         True, "procedural"),  # Has conceptual component
        ("Compare REBOA vs thoracotomy for hemorrhage control and describe the technique",
         True, "relationship"),  # Mixed relationship + procedural
    ],

    "decomposition_needed": [
        # Complex queries that should be decomposed
        ("How does blunt abdominal trauma differ from penetrating trauma in terms of diagnosis, management, and outcomes?",
         True, "relationship"),
        ("What are the indications for splenectomy vs splenic salvage, and what is the surgical technique for each?",
         True, "relationship"),
        ("Explain the physiology of shock, the classification systems, and the treatment algorithms for each type",
         True, "conceptual"),
    ],

    "domain_specific": [
        # Trauma surgery specific terminology
        ("What is the lethal triad in trauma?",
         True, "conceptual"),
        ("Explain FAST exam technique and interpretation",
         True, "procedural"),
        ("How does permissive hypotension work in trauma resuscitation?",
         True, "conceptual"),
        ("What are the zones of the neck and their surgical significance?",
         True, "conceptual"),
        ("Describe the Mattox maneuver for aortic exposure",
         True, "procedural"),
    ],

    "lookup_variations": [
        # Different lookup patterns
        ("Chapter 15 damage control surgery",
         True, "lookup"),
        ("Find the section on abdominal compartment syndrome",
         True, "lookup"),
        ("Where is traumatic brain injury management discussed?",
         True, "lookup"),
        ("Reference for Glasgow Coma Scale",
         True, "lookup"),
    ],

    "edge_cases": [
        # Ambiguous or tricky queries
        ("What is triage?",  # Simple but medical
         True, "conceptual"),
        ("How do surgeons make decisions?",  # Vague, somewhat medical
         True, "conceptual"),
        ("What equipment is needed for surgery?",  # Generic medical
         True, "conceptual"),
        ("Explain the history of trauma surgery",  # Historical, still medical
         True, "conceptual"),
    ],

    "non_medical_similar": [
        # Non-medical queries that might seem medical
        ("How does shock absorption work in car suspensions?",
         False, None),
        ("What is the trauma of war on soldiers' families?",  # Psychological, not medical procedure
         False, None),  # Could argue either way
        ("How do I perform a factory reset?",
         False, None),
        ("What is the protocol for fire evacuation?",
         False, None),
    ],
}


SYSTEM_PROMPT = """Classify medical queries for a trauma surgery textbook retrieval system.
Return JSON: {"medical":true/false,"intent":"conceptual"|"procedural"|"relationship"|"lookup"|null,"concepts":[]}

Intent types:
- conceptual: definitions, explanations, pathophysiology
- procedural: techniques, steps, how-to
- relationship: comparisons, effects, connections between concepts
- lookup: chapter/section references, finding specific content"""


def load_model():
    """Load Qwen2-1.5B MLX model."""
    print("Loading Qwen2-1.5B-Instruct-4bit (MLX)...")
    start = time.time()
    model, tokenizer = load("mlx-community/Qwen2-1.5B-Instruct-4bit")
    print(f"Loaded in {time.time() - start:.2f}s")
    return model, tokenizer


def classify_query(model, tokenizer, query: str) -> dict:
    """Classify a single query."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\nJSON:"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    start = time.perf_counter()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Parse response
    try:
        if "{" in response:
            json_str = response[response.find("{"):response.rfind("}") + 1]
            data = json.loads(json_str)
            return {
                "medical": data.get("medical", False),
                "intent": data.get("intent"),
                "concepts": data.get("concepts", []),
                "time_ms": elapsed_ms,
                "raw": response,
            }
    except:
        pass

    return {
        "medical": None,
        "intent": None,
        "concepts": [],
        "time_ms": elapsed_ms,
        "raw": response,
        "parse_error": True,
    }


def test_category(model, tokenizer, category: str, queries: list) -> dict:
    """Test a category of queries."""
    print(f"\n{'=' * 70}")
    print(f"Category: {category.upper()}")
    print("=" * 70)

    results = []
    correct_medical = 0
    correct_intent = 0
    total = len(queries)
    intent_total = 0
    times = []

    for query, expected_medical, expected_intent in queries:
        result = classify_query(model, tokenizer, query)
        times.append(result["time_ms"])

        # Check medical classification
        medical_match = result["medical"] == expected_medical
        if medical_match:
            correct_medical += 1

        # Check intent (only for medical queries)
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
        print(f"\n{result['time_ms']:6.1f}ms | {med_status} | {query[:60]}...")
        print(f"         Expected: {'Medical' if expected_medical else 'Non-med'} ({expected_intent})")
        print(f"         Got:      {'Medical' if result['medical'] else 'Non-med'} ({result['intent']})")
        if result.get("concepts"):
            print(f"         Concepts: {result['concepts']}")
        if not medical_match or (intent_match is False):
            print(f"         Raw: {result['raw'][:100]}...")

        results.append({
            "query": query,
            "expected_medical": expected_medical,
            "expected_intent": expected_intent,
            "result": result,
            "medical_correct": medical_match,
            "intent_correct": intent_match,
        })

    # Category summary
    med_acc = correct_medical / total * 100
    intent_acc = correct_intent / intent_total * 100 if intent_total > 0 else 0
    avg_time = sum(times) / len(times)

    print(f"\n--- {category} Summary ---")
    print(f"Medical: {correct_medical}/{total} ({med_acc:.0f}%)")
    print(f"Intent:  {correct_intent}/{intent_total} ({intent_acc:.0f}%)")
    print(f"Avg time: {avg_time:.1f}ms")

    return {
        "category": category,
        "medical_accuracy": med_acc,
        "intent_accuracy": intent_acc,
        "avg_time_ms": avg_time,
        "results": results,
    }


def main():
    """Run comprehensive complex query testing."""
    print("=" * 70)
    print("MedQuery - Complex Query Testing")
    print("Model: Qwen2-1.5B-Instruct-4bit (MLX)")
    print("=" * 70)

    model, tokenizer = load_model()

    # Warmup
    print("\nWarmup...")
    classify_query(model, tokenizer, "test query")

    # Test all categories
    all_results = []
    for category, queries in COMPLEX_QUERIES.items():
        result = test_category(model, tokenizer, category, queries)
        all_results.append(result)

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"{'Category':<25} {'Medical':<12} {'Intent':<12} {'Avg Time'}")
    print("-" * 70)

    total_medical_correct = 0
    total_medical = 0
    total_intent_correct = 0
    total_intent = 0
    all_times = []

    for r in all_results:
        print(f"{r['category']:<25} {r['medical_accuracy']:.0f}%{'':<7} {r['intent_accuracy']:.0f}%{'':<7} {r['avg_time_ms']:.1f}ms")

        for res in r["results"]:
            total_medical += 1
            if res["medical_correct"]:
                total_medical_correct += 1
            if res["expected_intent"]:
                total_intent += 1
                if res["intent_correct"]:
                    total_intent_correct += 1
            all_times.append(res["result"]["time_ms"])

    print("-" * 70)
    overall_med = total_medical_correct / total_medical * 100
    overall_intent = total_intent_correct / total_intent * 100 if total_intent > 0 else 0
    overall_time = sum(all_times) / len(all_times)

    print(f"{'OVERALL':<25} {overall_med:.0f}%{'':<7} {overall_intent:.0f}%{'':<7} {overall_time:.1f}ms")
    print(f"\nTotal queries: {total_medical}")
    print(f"Total time: {sum(all_times)/1000:.1f}s")

    # Identify problem areas
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    failures = []
    for r in all_results:
        for res in r["results"]:
            if not res["medical_correct"] or res["intent_correct"] is False:
                failures.append({
                    "category": r["category"],
                    "query": res["query"],
                    "expected": f"{'Medical' if res['expected_medical'] else 'Non-med'} ({res['expected_intent']})",
                    "got": f"{'Medical' if res['result']['medical'] else 'Non-med'} ({res['result']['intent']})",
                })

    if failures:
        print(f"\nFailed classifications ({len(failures)}):")
        for f in failures:
            print(f"\n  [{f['category']}]")
            print(f"  Query: {f['query'][:70]}...")
            print(f"  Expected: {f['expected']}")
            print(f"  Got: {f['got']}")
    else:
        print("\nAll classifications correct!")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if overall_med >= 90 and overall_intent >= 80:
        print("✓ Model is READY for production use")
    elif overall_med >= 80:
        print("~ Model is ACCEPTABLE but may need prompt tuning")
    else:
        print("✗ Model needs improvement or different approach")


if __name__ == "__main__":
    main()
