"""Evaluation framework for MedQuery models."""

from .schemas import (
    Entity,
    Relationship,
    SubQuery,
    QueryAnalysis,
    ValidationResult,
    ExpectedOutput,
    TestQuery,
)
from .validators import OutputValidator, ValidationMode, AccuracyScorer
from .parsers import JSONExtractor, extract_json
from .metrics import QueryResult, MetricsSnapshot, MetricsCollector
from .config import (
    BenchmarkConfig,
    ModelConfig,
    QUICK_TEST,
    STANDARD,
    THOROUGH,
    MLX_MODELS,
    API_MODELS,
)
from .runner import BenchmarkRunner, RunProgress, run_quick_benchmark
from .tui import BenchmarkTUI, print_final_report
from .reports import ReportGenerator

__all__ = [
    # Schemas
    "Entity",
    "Relationship",
    "SubQuery",
    "QueryAnalysis",
    "ValidationResult",
    "ExpectedOutput",
    "TestQuery",
    # Validation
    "OutputValidator",
    "ValidationMode",
    "AccuracyScorer",
    # Parsing
    "JSONExtractor",
    "extract_json",
    # Metrics
    "QueryResult",
    "MetricsSnapshot",
    "MetricsCollector",
    # Config
    "BenchmarkConfig",
    "ModelConfig",
    "QUICK_TEST",
    "STANDARD",
    "THOROUGH",
    "MLX_MODELS",
    "API_MODELS",
    # Runner
    "BenchmarkRunner",
    "RunProgress",
    "run_quick_benchmark",
    # TUI
    "BenchmarkTUI",
    "print_final_report",
    # Reports
    "ReportGenerator",
]
