#!/usr/bin/env python3
"""Explore single-stage query classification using Llama 3.1 8B Instruct.

After finding that Medicine LLM is unreliable for medical classification,
this script tests a simpler single-stage approach:
- One model (Llama 3.1 8B Instruct) does everything
- Classifies medical vs non-medical
- Determines intent type (conceptual/procedural/relationship/lookup)
- Extracts medical concepts

Benefits:
- Single model = faster, simpler
- Instruction-tuned = better at following prompts
- Good JSON output with proper prompting
"""

import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


# Test queries - mix of medical and non-medical
TEST_QUERIES = [
    # Medical - Conceptual
    ("What is hemorrhagic shock?", True, "conceptual"),
    ("Explain the physiology of sepsis", True, "conceptual"),
    ("What causes acute respiratory distress syndrome?", True, "conceptual"),

    # Medical - Procedural
    ("How to perform a thoracotomy?", True, "procedural"),
    ("Steps for laparoscopic cholecystectomy", True, "procedural"),
    ("What is the procedure for inserting a chest tube?", True, "procedural"),

    # Medical - Relationship
    ("What is the relationship between trauma and coagulopathy?", True, "relationship"),
    ("Compare open vs minimally invasive surgery", True, "relationship"),
    ("How does diabetes affect wound healing?", True, "relationship"),

    # Medical - Lookup
    ("Chapter 60 section on wound healing", True, "lookup"),
    ("Find information about appendectomy in chapter 12", True, "lookup"),

    # Non-medical
    ("What is the capital of France?", False, None),
    ("How do I make pasta carbonara?", False, None),
    ("Explain how a car engine works", False, None),
    ("What is machine learning?", False, None),
    ("Who wrote Romeo and Juliet?", False, None),
]


def get_model_path() -> Path:
    """Download Llama 3.1 8B Instruct if not cached."""
    cache_dir = Path.home() / ".cache" / "med-query" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Llama 3.1 8B Instruct - Q4_K_M quantization
    model_name = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    model_path = cache_dir / model_name

    if not model_path.exists():
        print(f"Downloading {model_name}...")
        print("This is ~4.9GB - may take a few minutes.")
        hf_hub_download(
            repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename=model_name,
            local_dir=cache_dir,
        )
        print(f"Downloaded to {model_path}")
    else:
        print(f"Using cached model: {model_path}")

    return model_path


def load_model(model_path: Path) -> Llama:
    """Load Llama 3.1 with chat format."""
    print("\nLoading Llama 3.1 8B Instruct...")
    start = time.time()

    llm = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1,
        chat_format="llama-3",
        verbose=False,
    )

    print(f"Model loaded in {time.time() - start:.2f}s")
    return llm


SYSTEM_PROMPT = """You are a query classification system for a medical textbook retrieval engine.

For each query, analyze and respond with JSON containing:
1. is_medical: boolean - Is this query related to medicine, healthcare, or medical topics?
2. intent: string - If medical, classify as one of:
   - "conceptual": Questions asking "what is", "explain", "describe" (definitions, explanations)
   - "procedural": Questions asking "how to", "steps for", "procedure" (processes, techniques)
   - "relationship": Questions about connections, comparisons, effects between concepts
   - "lookup": Direct references to chapters, sections, or specific locations
   - null: If not medical
3. medical_concepts: array - List of medical terms/concepts found (empty if not medical)
4. confidence: number 0-1 - Your confidence in the classification

Respond ONLY with valid JSON, no other text."""

USER_TEMPLATE = """Classify this query:
"{query}"

JSON:"""


def classify_query(llm: Llama, query: str) -> dict:
    """Classify a query using single-stage Llama 3.1."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(query=query)},
    ]

    start = time.time()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.1,
    )
    inference_time = time.time() - start

    response_text = response["choices"][0]["message"]["content"].strip()

    # Parse JSON
    try:
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            result["inference_time"] = inference_time
            result["raw_response"] = response_text
            return result
    except json.JSONDecodeError:
        pass

    return {
        "is_medical": None,
        "intent": None,
        "medical_concepts": [],
        "confidence": 0,
        "inference_time": inference_time,
        "raw_response": response_text,
        "parse_error": True,
    }


def test_classification(llm: Llama) -> list:
    """Test classification on all queries."""
    print("\n" + "=" * 70)
    print("Single-Stage Classification with Llama 3.1 8B Instruct")
    print("=" * 70)

    medical_correct = 0
    intent_correct = 0
    medical_total = 0
    intent_total = 0
    results = []
    total_time = 0

    for query, expected_medical, expected_intent in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print(f"Expected: {'Medical' if expected_medical else 'Non-medical'}", end="")
        if expected_intent:
            print(f" ({expected_intent})")
        else:
            print()

        result = classify_query(llm, query)
        total_time += result.get("inference_time", 0)

        # Check medical classification
        is_medical = result.get("is_medical")
        medical_match = is_medical == expected_medical
        if medical_match:
            medical_correct += 1
        medical_total += 1

        # Check intent classification (only for medical queries)
        intent = result.get("intent")
        intent_match = False
        if expected_medical and expected_intent:
            intent_total += 1
            if intent == expected_intent:
                intent_correct += 1
                intent_match = True

        status = "✓" if medical_match else "✗"
        print(f"Classified: {'Medical' if is_medical else 'Non-medical'} {status}", end="")
        if is_medical and intent:
            intent_status = "✓" if intent_match else "✗"
            print(f" ({intent}) {intent_status}")
        else:
            print()

        print(f"Concepts: {result.get('medical_concepts', [])}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"Time: {result.get('inference_time', 0):.2f}s")

        results.append({
            "query": query,
            "expected_medical": expected_medical,
            "expected_intent": expected_intent,
            "predicted_medical": is_medical,
            "predicted_intent": intent,
            "medical_correct": medical_match,
            "intent_correct": intent_match if expected_intent else None,
            "result": result,
        })

    # Summary
    medical_acc = medical_correct / medical_total * 100
    intent_acc = intent_correct / intent_total * 100 if intent_total > 0 else 0
    avg_time = total_time / len(TEST_QUERIES)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Medical Classification: {medical_correct}/{medical_total} ({medical_acc:.1f}%)")
    print(f"Intent Classification:  {intent_correct}/{intent_total} ({intent_acc:.1f}%)")
    print(f"Average inference time: {avg_time:.2f}s")
    print(f"Total time: {total_time:.1f}s")

    # Show errors
    medical_errors = [r for r in results if not r["medical_correct"]]
    intent_errors = [r for r in results if r["intent_correct"] is False]

    if medical_errors:
        print("\nMedical classification errors:")
        for r in medical_errors:
            print(f"  - {r['query']}")
            print(f"    Expected: {'Medical' if r['expected_medical'] else 'Non-medical'}")
            print(f"    Got: {'Medical' if r['predicted_medical'] else 'Non-medical'}")

    if intent_errors:
        print("\nIntent classification errors:")
        for r in intent_errors:
            print(f"  - {r['query']}")
            print(f"    Expected: {r['expected_intent']}")
            print(f"    Got: {r['predicted_intent']}")

    return results


def test_complex_queries(llm: Llama) -> None:
    """Test on more complex/ambiguous queries."""
    print("\n" + "=" * 70)
    print("Testing Complex Queries")
    print("=" * 70)

    complex_queries = [
        "How does trauma affect coagulation and what are the treatment steps?",
        "What is the relationship between Chapter 15's discussion of shock and the treatment protocols in Chapter 20?",
        "Compare the outcomes of laparoscopic vs open appendectomy in pediatric patients",
        "Explain why diabetic patients have delayed wound healing and what surgical techniques can help",
    ]

    for query in complex_queries:
        print(f"\nQuery: {query}")
        result = classify_query(llm, query)

        print(f"Medical: {result.get('is_medical')}")
        print(f"Intent: {result.get('intent')}")
        print(f"Concepts: {result.get('medical_concepts', [])}")
        print(f"Time: {result.get('inference_time', 0):.2f}s")


def main():
    """Run single-stage classification tests."""
    print("=" * 70)
    print("MedQuery - Single-Stage Classification with Llama 3.1")
    print("=" * 70)

    model_path = get_model_path()
    llm = load_model(model_path)

    test_classification(llm)
    test_complex_queries(llm)

    print("\n" + "=" * 70)
    print("Single-stage exploration complete!")
    print("=" * 70)
    print("\nKey questions:")
    print("1. Is accuracy good enough? (Target: >90% medical, >80% intent)")
    print("2. Is speed acceptable? (Target: <3s per query)")
    print("3. Do complex queries get reasonable classifications?")


if __name__ == "__main__":
    main()
