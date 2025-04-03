"""MLX backend for Apple Silicon inference.

Optimized for M1/M2/M3 Macs using the mlx-lm library.
"""

import asyncio
from typing import Any

from .base import BaseBackend, DEFAULT_SYSTEM_PROMPT
from ..logging import get_logger

logger = get_logger(__name__)


class MLXBackend(BaseBackend):
    """MLX backend for Apple Silicon."""

    def __init__(
        self,
        model_id: str,
        system_prompt: str | None = None,
        prompt_preset: str | None = None,
        adaptive_prompt: bool = False,
    ):
        """Initialize MLX backend.

        Args:
            model_id: HuggingFace model ID (e.g., "mlx-community/Qwen2-1.5B-Instruct-4bit")
            system_prompt: Optional custom system prompt
            prompt_preset: Preset name ("minimal", "standard", "comprehensive")
            adaptive_prompt: If True, select prompt based on query characteristics
        """
        super().__init__(
            model_id,
            system_prompt=system_prompt,
            prompt_preset=prompt_preset,
            adaptive_prompt=adaptive_prompt,
        )
        self._model = None
        self._tokenizer = None

    async def load(self) -> None:
        """Load model using mlx-lm."""
        if self._loaded:
            logger.debug("Model already loaded, skipping")
            return

        logger.info(f"Loading MLX model: {self._model_id}")

        from mlx_lm import load

        # Run sync load in executor
        loop = asyncio.get_event_loop()
        self._model, self._tokenizer = await loop.run_in_executor(
            None,
            lambda: load(self._model_id),
        )
        self._loaded = True
        logger.info(f"Model loaded successfully: {self._model_id}")

    async def unload(self) -> None:
        """Unload model to free memory."""
        if not self._loaded:
            logger.debug("Model not loaded, nothing to unload")
            return

        logger.info(f"Unloading model: {self._model_id}")
        self._model = None
        self._tokenizer = None
        self._loaded = False

        # Force garbage collection
        import gc
        gc.collect()
        logger.debug("Model unloaded and garbage collected")

    async def generate(
        self,
        query: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate response using MLX.

        Args:
            query: User query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            Generated response string
        """
        if not self._loaded:
            await self.load()

        from mlx_lm import generate

        logger.debug(f"Generating response for query: {query[:50]}...")

        # Build chat messages with adaptive prompt selection
        system_prompt = self.get_prompt_for_query(query)
        if self._adaptive_prompt:
            logger.debug(f"Adaptive prompt selected ({len(system_prompt)} chars)")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._build_prompt(query)},
        ]

        # Apply chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Run sync generate in executor
        # Note: mlx-lm uses sampler for temperature control
        # For deterministic output (temperature=0), we use the default greedy sampler
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            ),
        )

        self._total_requests += 1
        # Estimate tokens (rough)
        self._total_tokens += len(response.split()) * 1.3

        logger.debug(f"Generated response ({len(response)} chars): {response[:100]}...")
        return response

    async def generate_batch(
        self,
        queries: list[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate responses for multiple queries.

        Note: MLX doesn't have native batching, so we run sequentially.
        For better throughput, consider using async concurrency at a higher level.

        Args:
            queries: List of user queries
            max_tokens: Maximum tokens per query
            temperature: Sampling temperature

        Returns:
            List of generated responses
        """
        results = []
        for query in queries:
            response = await self.generate(
                query,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(response)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get MLX-specific stats."""
        stats = super().get_stats()
        stats["backend"] = "mlx"

        if self._loaded:
            try:
                import mlx.core as mx
                # Get memory stats if available
                stats["metal_memory"] = {
                    "peak_memory_gb": mx.metal.get_peak_memory() / 1e9,
                    "cache_memory_gb": mx.metal.get_cache_memory() / 1e9,
                }
            except Exception:
                pass

        return stats


# Register with factory
from .base import BackendFactory
BackendFactory.register("mlx", MLXBackend)
