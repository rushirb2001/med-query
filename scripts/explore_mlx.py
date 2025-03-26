#!/usr/bin/env python3
"""Explore MLX inference speed on Apple Silicon.

MLX is Apple's ML framework optimized for Apple Silicon.
This script compares MLX vs GGUF (llama.cpp) inference speed.

Setup (run manually first):
    pip install mlx-lm

MLX models are downloaded automatically from HuggingFace.
"""

import json
import time
import sys

# Check if mlx-lm is installed
try:
    from mlx_lm import load, generate
    print("MLX installed. Running MLX tests...")
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not installed. Run: pip install mlx-lm")
    print("Or: poetry add mlx-lm")


# Test queries
TEST_QUERIES = [
    ("What is hemorrhagic shock?", True, "conceptual"),
    ("How to perform a thoracotomy?", True, "procedural"),
    ("Compare open vs minimally invasive surgery", True, "relationship"),
    ("Chapter 60 section on wound healing", True, "lookup"),
    ("What is the capital of France?", False, None),
]


# MLX models to test (4-bit quantized for speed)
MLX_MODELS = {
    "qwen2-0.5b-4bit": "mlx-community/Qwen2-0.5B-Instruct-4bit",
    "qwen2-1.5b-4bit": "mlx-community/Qwen2-1.5B-Instruct-4bit",
    "phi3-mini-4bit": "mlx-community/Phi-3-mini-4k-instruct-4bit",
    "llama3.1-8b-4bit": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}


SYSTEM_PROMPT = """Classify query. Return JSON only:
{"medical":true/false,"intent":"conceptual"|"procedural"|"relationship"|"lookup"|null}"""


def test_mlx_model(model_name: str, model_path: str) -> dict:
    """Test MLX model inference speed."""
    print(f"\n{'=' * 60}")
    print(f"Testing MLX: {model_name}")
    print(f"Model: {model_path}")
    print("=" * 60)

    # Load model
    print("Loading model...")
    load_start = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - load_start
    print(f"Load time: {load_time:.2f}s")

    # Warmup
    print("Warmup...")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

    # Test queries
    times = []
    correct = 0

    for query, expected_medical, expected_intent in TEST_QUERIES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\nJSON:"},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        start = time.perf_counter()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=50,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        # Parse response
        try:
            if "{" in response:
                json_str = response[response.find("{"):response.rfind("}") + 1]
                data = json.loads(json_str)
                is_medical = data.get("medical", False)
            else:
                is_medical = None
        except:
            is_medical = None

        status = "✓" if is_medical == expected_medical else "✗"
        if is_medical == expected_medical:
            correct += 1

        print(f"  {elapsed_ms:6.1f}ms {status} | {query[:40]}")

    avg_time = sum(times) / len(times)
    accuracy = correct / len(TEST_QUERIES) * 100

    print(f"\nResults:")
    print(f"  Accuracy: {correct}/{len(TEST_QUERIES)} ({accuracy:.0f}%)")
    print(f"  Avg: {avg_time:.1f}ms | Min: {min(times):.1f}ms | Max: {max(times):.1f}ms")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "avg_ms": avg_time,
        "min_ms": min(times),
        "max_ms": max(times),
        "load_time": load_time,
    }


def test_gguf_comparison():
    """Test GGUF model for comparison."""
    from llama_cpp import Llama
    from pathlib import Path

    print(f"\n{'=' * 60}")
    print("Testing GGUF (llama.cpp) for comparison")
    print("=" * 60)

    model_path = Path.home() / ".cache" / "med-query" / "models" / "qwen2-0_5b-instruct-q4_k_m.gguf"

    if not model_path.exists():
        print(f"GGUF model not found: {model_path}")
        return None

    load_start = time.time()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=256,
        n_threads=4,
        n_gpu_layers=-1,
        chat_format="chatml",
        verbose=False,
    )
    load_time = time.time() - load_start
    print(f"Load time: {load_time:.2f}s")

    # Warmup
    messages = [{"role": "user", "content": "test"}]
    llm.create_chat_completion(messages=messages, max_tokens=10, temperature=0)

    # Test
    times = []
    for query, _, _ in TEST_QUERIES:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\nJSON:"},
        ]

        start = time.perf_counter()
        llm.create_chat_completion(messages=messages, max_tokens=50, temperature=0)
        times.append((time.perf_counter() - start) * 1000)

    print(f"  Avg: {sum(times)/len(times):.1f}ms | Min: {min(times):.1f}ms | Max: {max(times):.1f}ms")

    return {"avg_ms": sum(times) / len(times)}


def main():
    """Compare MLX vs GGUF inference speed."""
    print("=" * 60)
    print("MedQuery - MLX vs GGUF Speed Comparison")
    print("=" * 60)

    if not MLX_AVAILABLE:
        print("\nInstall MLX first:")
        print("  pip install mlx-lm")
        print("  # or")
        print("  poetry add mlx-lm")
        sys.exit(1)

    results = []

    # Test MLX models
    for name, path in MLX_MODELS.items():
        try:
            results.append(test_mlx_model(name, path))
        except Exception as e:
            print(f"Error testing {name}: {e}")

    # Test GGUF for comparison
    gguf_result = test_gguf_comparison()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: MLX vs GGUF")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':<10} {'Avg (ms)':<12} {'Backend'}")
    print("-" * 60)

    for r in results:
        print(f"{r['model']:<25} {r['accuracy']:.0f}%{'':<5} {r['avg_ms']:.1f}ms{'':<6} MLX")

    if gguf_result:
        print(f"{'qwen2-0.5b-gguf':<25} {'?':<10} {gguf_result['avg_ms']:.1f}ms{'':<6} GGUF")

    print("\nReference:")
    print("  Llama 3.1 8B GGUF: 3670ms avg, 100% accuracy")


if __name__ == "__main__":
    main()
