"""Base inference backend protocol and factory.

Defines the interface all backends must implement and
provides a factory for creating backends by type.
"""

from typing import Protocol, runtime_checkable, Any
from abc import abstractmethod


# Default system prompt for query classification
DEFAULT_SYSTEM_PROMPT = """You are a medical query classifier. Analyze queries and output JSON with:
- is_medical: boolean (true if medical domain)
- confidence: float 0.0-1.0
- primary_intent: "conceptual" | "procedural" | "relationship" | "lookup" | null
- entities: array of {text, type, cui} where type is condition|procedure|anatomy|process|concept|medication
- relationships: array of {source, target, type} where type is affects|causes|treats|indicates|compared_to

Output ONLY valid JSON, no explanation."""


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol for inference backends."""

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        ...

    async def load(self) -> None:
        """Load the model."""
        ...

    async def unload(self) -> None:
        """Unload the model to free memory."""
        ...

    async def generate(
        self,
        query: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Generate response for a single query.

        Args:
            query: User query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated response string
        """
        ...

    async def generate_batch(
        self,
        queries: list[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> list[str]:
        """Generate responses for multiple queries.

        Args:
            queries: List of user queries
            max_tokens: Maximum tokens to generate per query
            temperature: Sampling temperature

        Returns:
            List of generated response strings
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get runtime statistics.

        Returns:
            Dictionary with stats like tokens generated, memory usage, etc.
        """
        ...


class BaseBackend:
    """Base class with common backend functionality."""

    def __init__(
        self,
        model_id: str,
        system_prompt: str | None = None,
    ):
        self._model_id = model_id
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._loaded = False
        self._total_tokens = 0
        self._total_requests = 0

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "loaded": self._loaded,
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
        }

    def _build_prompt(self, query: str) -> str:
        """Build prompt with system message and query."""
        return f"Query: {query}\nJSON:"


class BackendFactory:
    """Factory for creating inference backends."""

    _backends: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, backend_class: type) -> None:
        """Register a backend class.

        Args:
            name: Backend name (e.g., "mlx", "llamacpp")
            backend_class: Backend class to register
        """
        cls._backends[name] = backend_class

    @classmethod
    def create(
        cls,
        backend_type: str,
        model_id: str,
        **kwargs,
    ) -> InferenceBackend:
        """Create a backend instance.

        Args:
            backend_type: Type of backend ("mlx", "llamacpp", "openai", "anthropic")
            model_id: Model identifier
            **kwargs: Additional backend-specific arguments

        Returns:
            Backend instance

        Raises:
            ValueError: If backend type is unknown
        """
        if backend_type not in cls._backends:
            # Try lazy import
            cls._lazy_import(backend_type)

        if backend_type not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(
                f"Unknown backend: {backend_type}. Available: {available}"
            )

        return cls._backends[backend_type](model_id, **kwargs)

    @classmethod
    def _lazy_import(cls, backend_type: str) -> None:
        """Lazily import and register a backend."""
        try:
            if backend_type == "mlx":
                from .mlx import MLXBackend
                cls.register("mlx", MLXBackend)
            elif backend_type == "llamacpp":
                from .llamacpp import LlamaCppBackend
                cls.register("llamacpp", LlamaCppBackend)
            elif backend_type in ("openai", "anthropic"):
                from .api import OpenAIBackend, AnthropicBackend
                cls.register("openai", OpenAIBackend)
                cls.register("anthropic", AnthropicBackend)
        except ImportError as e:
            raise ImportError(
                f"Could not import {backend_type} backend. "
                f"Make sure required dependencies are installed: {e}"
            )

    @classmethod
    def available_backends(cls) -> list[str]:
        """Get list of available backend types."""
        available = []
        for backend in ["mlx", "llamacpp", "openai", "anthropic"]:
            try:
                cls._lazy_import(backend)
                available.append(backend)
            except ImportError:
                pass
        return available
