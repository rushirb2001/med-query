"""MedQuery - LangGraph-based query routing and orchestration for medical textbook retrieval."""

__version__ = "0.1.0"

# Logging utilities for programmatic use
from .logging import setup_logging, get_logger, verbose, enable_debug

__all__ = [
    "__version__",
    "setup_logging",
    "get_logger",
    "verbose",
    "enable_debug",
]
