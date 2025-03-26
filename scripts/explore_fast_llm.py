#!/usr/bin/env python3
"""Explore fast LLM inference methods on Mac.

Goal: Get LLM-based query classification to milliseconds range.

Approaches tested:
1. Smaller models (Qwen2-0.5B, Phi-3-mini, TinyLlama)
2. Aggressive quantization (Q2_K, Q3_K)
3. llama.cpp optimizations (flash attention, batch size, context)
4. Prompt optimization (shorter prompts, fewer tokens)

For Mac: llama.cpp with Metal is already well-optimized.
Key is: smaller model + shorter prompt + minimal output tokens.
"""

import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama


# Minimal test set for speed testing
TEST_QUERIES = [
    ("What is hemorrhagic shock?", True, "conceptual"),
    ("How to perform a thoracotomy?", True, "procedural"),
    ("Compare open vs minimally invasive surgery", True, "relationship"),
    ("Chapter 60 section on wound healing", True, "lookup"),
    ("What is the capital of France?", False, None),
]


# Models to test (smallest and fastest)
MODELS = {
    "qwen2-0.5b-q4": {
        "repo": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "file": "qwen2-0_5b-instruct-q4_k_m.gguf",
        "size": "~400MB",
        "chat_format": "chatml",
    },
    "qwen2-0.5b-q2": {
        "repo": "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "file": "qwen2-0_5b-instruct-q2_k.gguf",
        "size": "~250MB",
        "chat_format": "chatml",
    },
    "tinyllama-q4": {
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size": "~670MB",
        "chat_format": "chatml",
    },
    "phi3-mini-q4": {
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "file": "Phi-3-mini-4k-instruct-q4.gguf",
        "size": "~2.2GB",
        "chat_format": "chatml",
    },
}


def get_model_path(model_key: str) -> Path:
    """Download model if needed."""
    model_info = MODELS[model_key]
    cache_dir = Path.home() / ".cache" / "med-query" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / model_info["file"]

    if not model_path.exists():
        print(f"Downloading {model_info['file']} ({model_info['size']})...")
        hf_hub_download(
            repo_id=model_info["repo"],
            filename=model_info["file"],
            local_dir=cache_dir,
        )
    return model_path


# Ultra-short prompt for speed (minimal tokens in, minimal out)
FAST_SYSTEM = """Classify query. JSON only: {"m":0/1,"i":"c/p/r/l/n"}
m=medical(1) or not(0)
i=conceptual/procedural/relationship/lookup/null"""

FAST_USER = """Q: {query}
J:"""


# Slightly longer but clearer prompt
MEDIUM_SYSTEM = """Classify medical queries. Return JSON:
{"medical":true/false,"intent":"conceptual"|"procedural"|"relationship"|"lookup"|null}"""

MEDIUM_USER = """Query: {query}
JSON:"""


def test_model(
    model_key: str,
    n_ctx: int = 256,
    max_tokens: int = 32,
    use_fast_prompt: bool = True,
) -> dict:
    """Test a model's classification speed."""
    model_info = MODELS[model_key]
    model_path = get_model_path(model_key)

    print(f"\n{'=' * 60}")
    print(f"Testing: {model_key}")
    print(f"Context: {n_ctx}, Max tokens: {max_tokens}")
    print("=" * 60)

    # Load model
    load_start = time.time()
    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=4,
        n_gpu_layers=-1,  # Full GPU/Metal
        flash_attn=True,  # Enable flash attention if available
        chat_format=model_info.get("chat_format"),
        verbose=False,
    )
    load_time = time.time() - load_start
    print(f"Load time: {load_time:.2f}s")

    # Select prompt
    if use_fast_prompt:
        system = FAST_SYSTEM
        user_template = FAST_USER
    else:
        system = MEDIUM_SYSTEM
        user_template = MEDIUM_USER

    # Warmup run
    print("Warmup...")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_template.format(query="test")},
    ]
    llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0)

    # Test queries
    times = []
    correct = 0

    for query, expected_medical, expected_intent in TEST_QUERIES:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_template.format(query=query)},
        ]

        start = time.perf_counter()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

        text = response["choices"][0]["message"]["content"].strip()

        # Try to parse response
        try:
            # Handle both prompt formats
            if "{" in text:
                json_str = text[text.find("{"):text.rfind("}") + 1]
                data = json.loads(json_str)
                # Handle short format
                if "m" in data:
                    is_medical = data.get("m") == 1
                else:
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
        "model": model_key,
        "accuracy": accuracy,
        "avg_ms": avg_time,
        "min_ms": min(times),
        "max_ms": max(times),
        "load_time": load_time,
    }


def test_context_sizes(model_key: str = "qwen2-0.5b-q4"):
    """Test how context size affects speed."""
    print("\n" + "=" * 60)
    print(f"Testing context sizes with {model_key}")
    print("=" * 60)

    model_path = get_model_path(model_key)

    for n_ctx in [128, 256, 512, 1024]:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=4,
            n_gpu_layers=-1,
            chat_format="chatml",
            verbose=False,
        )

        messages = [
            {"role": "system", "content": FAST_SYSTEM},
            {"role": "user", "content": FAST_USER.format(query="What is sepsis?")},
        ]

        # Warmup
        llm.create_chat_completion(messages=messages, max_tokens=32, temperature=0)

        # Measure
        times = []
        for _ in range(5):
            start = time.perf_counter()
            llm.create_chat_completion(messages=messages, max_tokens=32, temperature=0)
            times.append((time.perf_counter() - start) * 1000)

        print(f"  n_ctx={n_ctx}: avg={sum(times)/len(times):.1f}ms")


def main():
    """Test fast LLM inference approaches."""
    print("=" * 60)
    print("MedQuery - Fast LLM Inference Exploration")
    print("Goal: Sub-100ms classification")
    print("=" * 60)

    results = []

    # Test smallest model first (Qwen2 0.5B)
    results.append(test_model("qwen2-0.5b-q4", n_ctx=256, max_tokens=32))

    # Test even smaller quantization
    results.append(test_model("qwen2-0.5b-q2", n_ctx=256, max_tokens=32))

    # Test TinyLlama (slightly larger but well-optimized)
    results.append(test_model("tinyllama-q4", n_ctx=256, max_tokens=32))

    # Test context size impact
    test_context_sizes("qwen2-0.5b-q4")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Avg (ms)':<10} {'Min (ms)':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<20} {r['accuracy']:.0f}%{'':<5} {r['avg_ms']:.1f}{'':<5} {r['min_ms']:.1f}")

    print("\nComparison:")
    print("  Llama 3.1 8B: 100% accuracy, 3670ms avg")
    print("\nTrade-off: Speed vs Accuracy")
    print("  - For async requests, sub-100ms may be achievable with small models")
    print("  - May need hybrid: fast model for simple cases, larger for complex")


if __name__ == "__main__":
    main()
