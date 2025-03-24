#!/usr/bin/env python3
"""Explore llama-cpp-python setup and verify it works on this machine.

This script tests:
1. llama-cpp-python installation
2. Model downloading from HuggingFace
3. Basic inference
4. GPU/Metal acceleration
5. Inference speed benchmarking

Using a small model (TinyLlama) for quick testing before downloading
larger models for actual classification.
"""

import time
from pathlib import Path

from huggingface_hub import hf_hub_download


def get_model_path() -> Path:
    """Download a small test model if not already cached."""
    cache_dir = Path.home() / ".cache" / "med-query" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # TinyLlama 1.1B - small enough for quick testing
    # Q4_K_M quantization for good quality/size balance
    model_name = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_path = cache_dir / model_name

    if not model_path.exists():
        print(f"Downloading {model_name}...")
        print("This may take a few minutes on first run.")
        hf_hub_download(
            repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            filename=model_name,
            local_dir=cache_dir,
        )
        print(f"Downloaded to {model_path}")
    else:
        print(f"Using cached model: {model_path}")

    return model_path


def test_basic_inference(model_path: Path) -> None:
    """Test basic inference with the model."""
    from llama_cpp import Llama

    print("\n" + "=" * 60)
    print("Testing Basic Inference")
    print("=" * 60)

    print("\nLoading model...")
    start = time.time()

    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,  # Small context for testing
        n_threads=4,
        n_gpu_layers=-1,  # Use all GPU layers (Metal on Mac)
        verbose=False,
    )

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Test simple completion
    prompt = "What is the capital of France? Answer in one word:"

    print(f"\nPrompt: {prompt}")
    print("Generating...")

    start = time.time()
    output = llm(
        prompt,
        max_tokens=32,
        temperature=0.1,
        stop=["\n"],
        echo=False,
    )
    inference_time = time.time() - start

    response = output["choices"][0]["text"].strip()
    print(f"Response: {response}")
    print(f"Inference time: {inference_time:.2f}s")


def test_chat_completion(model_path: Path) -> None:
    """Test chat completion format."""
    from llama_cpp import Llama

    print("\n" + "=" * 60)
    print("Testing Chat Completion")
    print("=" * 60)

    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=-1,
        chat_format="chatml",  # TinyLlama uses ChatML format
        verbose=False,
    )

    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": "What is hemorrhagic shock? Answer briefly."},
    ]

    print(f"\nMessages: {messages}")
    print("Generating...")

    start = time.time()
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=100,
        temperature=0.1,
    )
    inference_time = time.time() - start

    answer = response["choices"][0]["message"]["content"]
    print(f"Response: {answer}")
    print(f"Inference time: {inference_time:.2f}s")


def benchmark_inference(model_path: Path, num_runs: int = 5) -> None:
    """Benchmark inference speed."""
    from llama_cpp import Llama

    print("\n" + "=" * 60)
    print(f"Benchmarking Inference ({num_runs} runs)")
    print("=" * 60)

    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=-1,
        verbose=False,
    )

    prompt = "Is this a medical question? Answer yes or no:"
    times = []

    for i in range(num_runs):
        start = time.time()
        llm(prompt, max_tokens=10, temperature=0.1, echo=False)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nResults:")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Min: {min_time:.3f}s")
    print(f"  Max: {max_time:.3f}s")


def check_gpu_support() -> None:
    """Check GPU/Metal support."""
    print("\n" + "=" * 60)
    print("Checking GPU Support")
    print("=" * 60)

    try:
        from llama_cpp import llama_supports_gpu_offload

        if llama_supports_gpu_offload():
            print("GPU offloading is SUPPORTED")
            print("(Metal on macOS, CUDA on Linux/Windows)")
        else:
            print("GPU offloading is NOT supported")
            print("Running on CPU only")
    except ImportError:
        print("Could not check GPU support (older llama-cpp-python version)")
        print("Assuming GPU support based on n_gpu_layers parameter")


def main():
    """Run all setup tests."""
    print("=" * 60)
    print("MedQuery - LLM Setup Exploration")
    print("=" * 60)

    # Check GPU support
    check_gpu_support()

    # Download/locate model
    model_path = get_model_path()

    # Run tests
    test_basic_inference(model_path)
    test_chat_completion(model_path)
    benchmark_inference(model_path)

    print("\n" + "=" * 60)
    print("Setup verification complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download Medicine LLM for medical classification")
    print("2. Download Llama 3.1 8B Instruct for intent classification")
    print("3. Run explore_medical_classify.py")


if __name__ == "__main__":
    main()
