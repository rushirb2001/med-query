"""Base inference backend protocol and factory.

Defines the interface all backends must implement and
provides a factory for creating backends by type.
"""

from typing import Protocol, runtime_checkable, Any
from abc import abstractmethod

from .prompts import PROMPT_PRESETS, get_prompt, PromptPreset, PromptSelector
from ..logging import get_logger

logger = get_logger(__name__)


# Default system prompt - now uses "standard" preset with intent rules
DEFAULT_SYSTEM_PROMPT = PROMPT_PRESETS["standard"]

# Global prompt selector for adaptive prompt selection
_prompt_selector = PromptSelector(default_preset="standard")


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
        prompt_preset: PromptPreset | None = None,
        adaptive_prompt: bool = False,
    ):
        """Initialize backend.

        Args:
            model_id: Model identifier
            system_prompt: Explicit system prompt (highest priority)
            prompt_preset: Preset name ("minimal", "standard", "comprehensive")
            adaptive_prompt: If True, select prompt based on query characteristics
        """
        self._model_id = model_id
        self._adaptive_prompt = adaptive_prompt
        self._prompt_selector = PromptSelector(
            default_preset=prompt_preset or "standard"
        )

        # Priority: explicit system_prompt > prompt_preset > default
        if system_prompt is not None:
            self._base_prompt = system_prompt
            logger.debug(f"Using custom system prompt ({len(system_prompt)} chars)")
        elif prompt_preset is not None:
            self._base_prompt = get_prompt(prompt_preset)
            logger.debug(f"Using prompt preset: {prompt_preset}")
        else:
            self._base_prompt = DEFAULT_SYSTEM_PROMPT
            logger.debug("Using default system prompt (standard)")

        # For backwards compatibility
        self.system_prompt = self._base_prompt

        self._loaded = False
        self._total_tokens = 0
        self._total_requests = 0

        logger.info(f"Initialized backend: model={model_id}, adaptive={adaptive_prompt}")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_prompt_for_query(self, query: str) -> str:
        """Get the appropriate system prompt for a query.

        Args:
            query: The user query

        Returns:
            System prompt string (adaptive or static based on config)
        """
        if self._adaptive_prompt:
            return self._prompt_selector.select(query)
        return self._base_prompt

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_id": self._model_id,
            "loaded": self._loaded,
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
            "adaptive_prompt": self._adaptive_prompt,
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
        logger.debug(f"Creating backend: type={backend_type}, model={model_id}")

        if backend_type not in cls._backends:
            # Try lazy import
            logger.debug(f"Lazy importing backend: {backend_type}")
            cls._lazy_import(backend_type)

        if backend_type not in cls._backends:
            available = list(cls._backends.keys())
            logger.error(f"Unknown backend: {backend_type}. Available: {available}")
            raise ValueError(
                f"Unknown backend: {backend_type}. Available: {available}"
            )

        logger.info(f"Created {backend_type} backend for {model_id}")
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
