#!/usr/bin/env python3
"""Explore Stage 1: Medical domain classification using Medicine LLM.

This script tests:
1. Loading Medicine LLM (domain-specific model)
2. Classifying queries as medical vs non-medical
3. Extracting medical concepts from queries
4. Testing different prompt formats
5. Evaluating classification accuracy

Goal: Determine if a medical-trained LLM can reliably identify
medical queries and extract relevant concepts.
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

    # Non-medical (should be classified as non-medical)
    ("What is the capital of France?", False, None),
    ("How do I make pasta carbonara?", False, None),
    ("Explain how a car engine works", False, None),
    ("What is machine learning?", False, None),
    ("Who wrote Romeo and Juliet?", False, None),
]


def get_model_path() -> Path:
    """Download Medicine LLM if not already cached."""
    cache_dir = Path.home() / ".cache" / "med-query" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Medicine LLM - medical domain specific
    model_name = "medicine-llm-13b.Q4_K_M.gguf"
    model_path = cache_dir / model_name

    if not model_path.exists():
        print(f"Downloading {model_name}...")
        print("This is a 7.8GB model - may take several minutes.")
        hf_hub_download(
            repo_id="TheBloke/medicine-LLM-13B-GGUF",
            filename=model_name,
            local_dir=cache_dir,
        )
        print(f"Downloaded to {model_path}")
    else:
        print(f"Using cached model: {model_path}")

    return model_path


def load_model(model_path: Path) -> Llama:
    """Load the model with appropriate settings."""
    print("\nLoading Medicine LLM...")
    start = time.time()

    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=-1,  # All layers on GPU/Metal
        verbose=False,
    )

    print(f"Model loaded in {time.time() - start:.2f}s")
    return llm


# Prompt template for medical classification
CLASSIFICATION_PROMPT = """Analyze the following query and determine:
1. Is this a medical/healthcare related question? (yes/no)
2. If medical, what medical concepts or terms are mentioned?
3. Rate the medical relevance (high/medium/low/none)

Query: {query}

Respond in JSON format:
{{"is_medical": true/false, "medical_concepts": ["concept1", "concept2"], "relevance": "high/medium/low/none", "reasoning": "brief explanation"}}

JSON Response:"""


def classify_query(llm: Llama, query: str) -> dict:
    """Classify a query using the medical LLM."""
    prompt = CLASSIFICATION_PROMPT.format(query=query)

    start = time.time()
    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.1,
        stop=["\n\n", "Query:"],
        echo=False,
    )
    inference_time = time.time() - start

    response_text = response["choices"][0]["text"].strip()

    # Try to parse JSON
    try:
        # Find JSON in response
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

    # Fallback: return raw response
    return {
        "is_medical": None,
        "medical_concepts": [],
        "relevance": "unknown",
        "reasoning": "Failed to parse JSON",
        "inference_time": inference_time,
        "raw_response": response_text,
    }


def test_classification(llm: Llama) -> None:
    """Test classification on all test queries."""
    print("\n" + "=" * 70)
    print("Testing Medical Classification")
    print("=" * 70)

    correct = 0
    total = len(TEST_QUERIES)
    results = []

    for query, expected_medical, expected_type in TEST_QUERIES:
        print(f"\nQuery: {query}")
        print(f"Expected: {'Medical' if expected_medical else 'Non-medical'}")

        result = classify_query(llm, query)

        # Determine if classification is correct
        is_medical = result.get("is_medical")
        if isinstance(is_medical, str):
            is_medical = is_medical.lower() in ("yes", "true")

        is_correct = is_medical == expected_medical
        if is_correct:
            correct += 1

        print(f"Classified: {'Medical' if is_medical else 'Non-medical'} {'✓' if is_correct else '✗'}")
        print(f"Relevance: {result.get('relevance', 'N/A')}")
        print(f"Concepts: {result.get('medical_concepts', [])}")
        print(f"Time: {result.get('inference_time', 0):.2f}s")

        results.append({
            "query": query,
            "expected_medical": expected_medical,
            "predicted_medical": is_medical,
            "correct": is_correct,
            "result": result,
        })

    # Summary
    accuracy = correct / total * 100
    print("\n" + "=" * 70)
    print(f"Classification Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("=" * 70)

    # Show misclassifications
    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        print("\nMisclassified queries:")
        for r in misclassified:
            print(f"  - {r['query']}")
            print(f"    Expected: {'Medical' if r['expected_medical'] else 'Non-medical'}")
            print(f"    Got: {'Medical' if r['predicted_medical'] else 'Non-medical'}")

    return results


def test_concept_extraction(llm: Llama) -> None:
    """Test if the model extracts meaningful medical concepts."""
    print("\n" + "=" * 70)
    print("Testing Medical Concept Extraction")
    print("=" * 70)

    test_cases = [
        ("What is hemorrhagic shock and how does it affect coagulation?",
         ["hemorrhagic shock", "coagulation"]),
        ("Explain the relationship between diabetes mellitus and peripheral neuropathy",
         ["diabetes mellitus", "peripheral neuropathy"]),
        ("How do NSAIDs affect platelet function?",
         ["NSAIDs", "platelet function"]),
    ]

    for query, expected_concepts in test_cases:
        print(f"\nQuery: {query}")
        print(f"Expected concepts: {expected_concepts}")

        result = classify_query(llm, query)
        extracted = result.get("medical_concepts", [])
        print(f"Extracted concepts: {extracted}")

        # Check overlap
        expected_lower = [c.lower() for c in expected_concepts]
        extracted_lower = [c.lower() for c in extracted]

        found = sum(1 for e in expected_lower if any(e in x or x in e for x in extracted_lower))
        print(f"Concept match: {found}/{len(expected_concepts)}")


def test_prompt_variants(llm: Llama) -> None:
    """Test different prompt formats to find what works best."""
    print("\n" + "=" * 70)
    print("Testing Prompt Variants")
    print("=" * 70)

    test_query = "What is hemorrhagic shock?"

    prompts = {
        "json_explicit": """Is this query medical-related? Respond with JSON only.
Query: {query}
{{"is_medical": true/false, "concepts": [], "relevance": "high/medium/low/none"}}""",

        "simple": """Is this a medical question? Answer yes or no.
Query: {query}
Answer:""",

        "medical_expert": """You are a medical expert. Analyze this query.
Query: {query}
Is this medical-related (yes/no):
Medical concepts found:
Relevance (high/medium/low/none):""",

        "structured": """### Task: Medical Query Classification
### Query: {query}
### Analysis:
- Medical related: [yes/no]
- Key medical terms: [list]
- Relevance level: [high/medium/low/none]
### Response:""",
    }

    for name, template in prompts.items():
        print(f"\n--- Prompt: {name} ---")
        prompt = template.format(query=test_query)

        start = time.time()
        response = llm(
            prompt,
            max_tokens=100,
            temperature=0.1,
            stop=["\n\n", "###", "Query:"],
            echo=False,
        )
        elapsed = time.time() - start

        response_text = response["choices"][0]["text"].strip()
        print(f"Response: {response_text[:200]}")
        print(f"Time: {elapsed:.2f}s")


def main():
    """Run all medical classification tests."""
    print("=" * 70)
    print("MedQuery - Medical Classification Exploration (Stage 1)")
    print("=" * 70)

    # Download/load model
    model_path = get_model_path()
    llm = load_model(model_path)

    # Run tests
    test_classification(llm)
    test_concept_extraction(llm)
    test_prompt_variants(llm)

    print("\n" + "=" * 70)
    print("Medical classification exploration complete!")
    print("=" * 70)
    print("\nFindings to consider:")
    print("1. Does the model reliably distinguish medical vs non-medical?")
    print("2. Are extracted concepts accurate and useful?")
    print("3. Which prompt format works best?")
    print("4. Is inference speed acceptable for the two-stage pipeline?")


if __name__ == "__main__":
    main()
