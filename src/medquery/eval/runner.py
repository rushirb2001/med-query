"""Benchmark runner for evaluating query classification models.

Orchestrates:
- Dataset loading and sampling
- Model inference (warmup + evaluation)
- Metrics collection
- Progress tracking
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable

from .schemas import TestQuery, ExpectedOutput, QueryAnalysis, ValidationResult
from .validators import OutputValidator
from .metrics import QueryResult, MetricsCollector, MetricsSnapshot
from .config import BenchmarkConfig


@dataclass
class RunProgress:
    """Progress update for TUI/logging."""

    phase: str  # "warmup", "evaluation", "computing"
    current: int
    total: int
    current_query: str | None = None
    latency_ms: float | None = None
    is_valid: bool | None = None
    error: str | None = None


class BenchmarkRunner:
    """Run benchmarks on query classification models."""

    def __init__(
        self,
        config: BenchmarkConfig,
        progress_callback: Callable[[RunProgress], None] | None = None,
    ):
        """Initialize runner.

        Args:
            config: Benchmark configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.progress_callback = progress_callback or (lambda _: None)

        self._backend = None
        self._validator = OutputValidator(mode=config.validation_mode)
        self._collector = MetricsCollector()

        self._queries: list[TestQuery] = []
        self._raw_outputs: list[dict] = []

    async def setup(self) -> None:
        """Initialize backend and load dataset."""
        from ..backends.base import BackendFactory

        # Create backend
        self._backend = BackendFactory.create(
            backend_type=self.config.backend,
            model_id=self.config.model_id,
        )

        # Load model
        await self._backend.load()

        # Load dataset
        self._queries = self._load_dataset()

    async def teardown(self) -> None:
        """Cleanup resources."""
        if self._backend:
            await self._backend.unload()

    def _load_dataset(self) -> list[TestQuery]:
        """Load and filter dataset from config path."""
        queries = []

        # Check if it's a directory with multiple files or single file
        dataset_path = self.config.dataset_path
        if dataset_path.is_dir():
            json_files = list(dataset_path.glob("*.json"))
        else:
            json_files = [dataset_path]

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            # Handle both list format and dict with 'queries' key
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "queries" in data:
                items = data["queries"]
            else:
                continue

            for item in items:
                try:
                    query = TestQuery.model_validate(item)

                    # Filter by category if specified
                    if self.config.categories:
                        if query.category not in self.config.categories:
                            continue

                    queries.append(query)
                except Exception:
                    continue

        # Sample if needed
        if self.config.sample_size and len(queries) > self.config.sample_size:
            random.seed(self.config.seed)
            queries = random.sample(queries, self.config.sample_size)

        return queries

    async def run_warmup(self) -> None:
        """Run warmup queries to stabilize performance."""
        if not self._queries:
            return

        warmup_queries = self._queries[: self.config.warmup_queries]

        for i, test_query in enumerate(warmup_queries):
            self.progress_callback(
                RunProgress(
                    phase="warmup",
                    current=i + 1,
                    total=len(warmup_queries),
                    current_query=test_query.query[:50] + "..."
                    if len(test_query.query) > 50
                    else test_query.query,
                )
            )

            try:
                await self._backend.generate(
                    test_query.query,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception:
                pass  # Ignore warmup errors

    async def run_evaluation(self) -> AsyncIterator[QueryResult]:
        """Run evaluation and yield results.

        Yields:
            QueryResult for each evaluated query
        """
        for i, test_query in enumerate(self._queries):
            self.progress_callback(
                RunProgress(
                    phase="evaluation",
                    current=i + 1,
                    total=len(self._queries),
                    current_query=test_query.query[:50] + "..."
                    if len(test_query.query) > 50
                    else test_query.query,
                )
            )

            start_time = time.perf_counter()
            error = None
            raw_output = ""

            try:
                raw_output = await self._backend.generate(
                    test_query.query,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                error = str(e)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Parse and validate output
            predicted: QueryAnalysis | None = None
            validation_errors: list[str] = []
            validation_warnings: list[str] = []

            if raw_output and not error:
                # Use validator for full parsing and validation
                validation = self._validator.validate(raw_output)
                predicted = validation.parsed
                validation_errors = validation.errors.copy()
                validation_warnings = validation.warnings.copy()
            else:
                # Create validation result for error case
                validation_errors = [error] if error else ["Empty output"]
                validation = ValidationResult(
                    valid=False,
                    parsed=None,
                    errors=validation_errors,
                    warnings=[],
                    raw_output=raw_output,
                    parse_time_ms=0,
                )

            # Create result
            result = QueryResult(
                query_id=test_query.id,
                query=test_query.query,
                category=test_query.category,
                level=test_query.level,
                expected=test_query.expected,
                predicted=predicted,
                validation=validation,
                latency_ms=latency_ms,
            )

            # Store raw output if configured
            if self.config.save_raw_outputs:
                self._raw_outputs.append(
                    {
                        "query_id": test_query.id,
                        "query": test_query.query,
                        "raw_output": raw_output,
                        "parsed": predicted.model_dump() if predicted else None,
                        "error": error,
                    }
                )

            # Update progress callback
            self.progress_callback(
                RunProgress(
                    phase="evaluation",
                    current=i + 1,
                    total=len(self._queries),
                    current_query=test_query.query[:50] + "...",
                    latency_ms=latency_ms,
                    is_valid=result.is_valid,
                    error=error,
                )
            )

            yield result

    async def run(self) -> MetricsSnapshot:
        """Run complete benchmark and return metrics.

        Returns:
            Aggregated metrics from the run
        """
        await self.setup()

        try:
            # Warmup
            await self.run_warmup()

            # Clear collector for fresh run
            self._collector.clear()

            # Run evaluation
            async for result in self.run_evaluation():
                self._collector.add(result)

            # Compute metrics
            self.progress_callback(
                RunProgress(
                    phase="computing",
                    current=1,
                    total=1,
                )
            )

            metrics = self._collector.compute()

            # Save results if configured
            if self.config.save_predictions or self.config.save_raw_outputs:
                self._save_results(metrics)

            return metrics

        finally:
            await self.teardown()

    async def run_multiple(self) -> list[MetricsSnapshot]:
        """Run multiple evaluation rounds.

        Returns:
            List of metrics from each run
        """
        all_metrics = []

        for run_idx in range(self.config.num_runs):
            # Reset for new run
            self._collector.clear()
            self._raw_outputs.clear()

            # Re-shuffle if multiple runs
            if run_idx > 0:
                random.seed(self.config.seed + run_idx)
                random.shuffle(self._queries)

            # Run evaluation (skip warmup after first run)
            if run_idx == 0:
                await self.run_warmup()

            async for result in self.run_evaluation():
                self._collector.add(result)

            metrics = self._collector.compute()
            all_metrics.append(metrics)

        return all_metrics

    def _save_results(self, metrics: MetricsSnapshot) -> None:
        """Save results to output directory."""
        results_dir = self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(results_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save metrics
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Save predictions
        if self.config.save_predictions:
            predictions = [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "category": r.category,
                    "level": r.level,
                    "expected": r.expected.model_dump(),
                    "predicted": r.predicted.model_dump() if r.predicted else None,
                    "validation": {
                        "valid": r.validation.valid,
                        "errors": r.validation.errors,
                    },
                    "latency_ms": r.latency_ms,
                }
                for r in self._collector.results
            ]
            with open(results_dir / "predictions.json", "w") as f:
                json.dump(predictions, f, indent=2)

        # Save raw outputs
        if self.config.save_raw_outputs and self._raw_outputs:
            with open(results_dir / "raw_outputs.json", "w") as f:
                json.dump(self._raw_outputs, f, indent=2)

    def get_collector(self) -> MetricsCollector:
        """Get metrics collector for external analysis."""
        return self._collector


async def run_quick_benchmark(
    model_id: str = "mlx-community/Qwen2-1.5B-Instruct-4bit",
    dataset_path: str | Path = "scripts/data",
    sample_size: int = 100,
) -> MetricsSnapshot:
    """Run a quick benchmark with minimal configuration.

    Args:
        model_id: Model to evaluate
        dataset_path: Path to dataset
        sample_size: Number of queries to test

    Returns:
        Metrics snapshot
    """
    from .config import QUICK_TEST

    config = BenchmarkConfig(
        dataset_path=dataset_path,
        model_id=model_id,
        sample_size=sample_size,
        num_runs=1,
        warmup_queries=5,
        enable_tui=False,
    )

    runner = BenchmarkRunner(config)
    return await runner.run()
