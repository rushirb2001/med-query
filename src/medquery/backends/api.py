"""API backends for cloud inference (OpenAI, Anthropic).

Future expansion for comparing local models against API baselines.
"""

import asyncio
from typing import Any

from .base import BaseBackend, DEFAULT_SYSTEM_PROMPT, BackendFactory
from ..logging import get_logger

logger = get_logger(__name__)


class OpenAIBackend(BaseBackend):
    """OpenAI API backend."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize OpenAI backend.

        Args:
            model_id: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            system_prompt: Optional custom system prompt
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        super().__init__(model_id, system_prompt)
        self._api_key = api_key
        self._client = None

    async def load(self) -> None:
        """Initialize OpenAI client."""
        if self._loaded:
            return

        logger.info(f"Loading OpenAI backend: {self._model_id}")
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self._client = AsyncOpenAI(api_key=self._api_key)
        self._loaded = True
        logger.debug("OpenAI client initialized")

    async def unload(self) -> None:
        """Close client."""
        if not self._loaded:
            return

        if self._client:
            await self._client.close()
        self._client = None
        self._loaded = False

    async def generate(
        self,
        query: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate response using OpenAI API.

        Args:
            query: User query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response string
        """
        if not self._loaded:
            await self.load()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_prompt(query)},
        ]

        response = await self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        self._total_requests += 1
        self._total_tokens += response.usage.total_tokens if response.usage else 0

        return response.choices[0].message.content or ""

    async def generate_batch(
        self,
        queries: list[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate responses for multiple queries concurrently.

        Args:
            queries: List of user queries
            max_tokens: Maximum tokens per query
            temperature: Sampling temperature

        Returns:
            List of generated responses
        """
        tasks = [
            self.generate(query, max_tokens, temperature)
            for query in queries
        ]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        """Get OpenAI-specific stats."""
        stats = super().get_stats()
        stats["backend"] = "openai"
        return stats


class AnthropicBackend(BaseBackend):
    """Anthropic API backend."""

    def __init__(
        self,
        model_id: str = "claude-3-haiku-20240307",
        system_prompt: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize Anthropic backend.

        Args:
            model_id: Model name (e.g., "claude-3-haiku-20240307")
            system_prompt: Optional custom system prompt
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        super().__init__(model_id, system_prompt)
        self._api_key = api_key
        self._client = None

    async def load(self) -> None:
        """Initialize Anthropic client."""
        if self._loaded:
            return

        logger.info(f"Loading Anthropic backend: {self._model_id}")
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self._client = AsyncAnthropic(api_key=self._api_key)
        self._loaded = True
        logger.debug("Anthropic client initialized")

    async def unload(self) -> None:
        """Close client."""
        if not self._loaded:
            return

        self._client = None
        self._loaded = False

    async def generate(
        self,
        query: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate response using Anthropic API.

        Args:
            query: User query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response string
        """
        if not self._loaded:
            await self.load()

        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": self._build_prompt(query)},
            ],
            temperature=temperature if temperature > 0 else None,
        )

        self._total_requests += 1
        self._total_tokens += response.usage.input_tokens + response.usage.output_tokens

        return response.content[0].text if response.content else ""

    async def generate_batch(
        self,
        queries: list[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate responses for multiple queries concurrently.

        Args:
            queries: List of user queries
            max_tokens: Maximum tokens per query
            temperature: Sampling temperature

        Returns:
            List of generated responses
        """
        tasks = [
            self.generate(query, max_tokens, temperature)
            for query in queries
        ]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        """Get Anthropic-specific stats."""
        stats = super().get_stats()
        stats["backend"] = "anthropic"
        return stats


# Register with factory
BackendFactory.register("openai", OpenAIBackend)
BackendFactory.register("anthropic", AnthropicBackend)
