"""CLI entry point for MedQuery benchmark evaluation.

Usage:
    medquery-eval benchmark --model mlx-community/Qwen2-1.5B-Instruct-4bit
    medquery-eval quick-test --sample-size 50
    medquery-eval report results/20250327_123456
"""

import asyncio
from pathlib import Path

import click
from rich.console import Console

from .config import BenchmarkConfig, ValidationMode, MLX_MODELS
from .runner import BenchmarkRunner
from .tui import BenchmarkTUI, print_final_report
from .reports import ReportGenerator
from ..logging import setup_logging, get_logger

logger = get_logger(__name__)
console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="medquery-eval")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output (INFO level logging)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output (DEBUG level logging)",
)
@click.option(
    "--log-file",
    type=click.Path(),
    help="Write logs to file",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool, log_file: str | None):
    """MedQuery Benchmark Evaluation CLI."""
    # Ensure context object exists
    ctx.ensure_object(dict)

    # Setup logging based on flags
    setup_logging(verbose=verbose, debug=debug, log_file=log_file)

    if debug:
        logger.debug("Debug logging enabled")
    elif verbose:
        logger.info("Verbose logging enabled")


@cli.command()
@click.option(
    "--model",
    "-m",
    default="mlx-community/Qwen2-1.5B-Instruct-4bit",
    help="Model ID to evaluate",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["mlx", "llamacpp", "openai", "anthropic"]),
    default="mlx",
    help="Inference backend",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    default="scripts/data",
    help="Path to dataset directory or file",
)
@click.option(
    "--sample-size",
    "-n",
    type=int,
    default=None,
    help="Number of queries to sample (default: all)",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for evaluation",
)
@click.option(
    "--num-runs",
    type=int,
    default=1,
    help="Number of evaluation runs",
)
@click.option(
    "--warmup",
    type=int,
    default=10,
    help="Number of warmup queries",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="results",
    help="Output directory for results",
)
@click.option(
    "--validation-mode",
    type=click.Choice(["strict", "lenient", "partial"]),
    default="lenient",
    help="Validation strictness",
)
@click.option(
    "--no-tui",
    is_flag=True,
    help="Disable TUI dashboard",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Minimal output",
)
@click.option(
    "--save-raw",
    is_flag=True,
    help="Save raw model outputs",
)
@click.option(
    "--prompt-preset",
    "-p",
    type=click.Choice(["minimal", "standard", "comprehensive"]),
    default=None,
    help="Prompt preset to use (default: standard)",
)
@click.option(
    "--adaptive-prompt",
    is_flag=True,
    help="Enable adaptive prompt selection per query",
)
def benchmark(
    model: str,
    backend: str,
    dataset: str,
    sample_size: int | None,
    batch_size: int,
    num_runs: int,
    warmup: int,
    output: str,
    validation_mode: str,
    no_tui: bool,
    quiet: bool,
    save_raw: bool,
    prompt_preset: str | None,
    adaptive_prompt: bool,
):
    """Run full benchmark evaluation."""
    config = BenchmarkConfig(
        dataset_path=dataset,
        model_id=model,
        backend=backend,
        sample_size=sample_size,
        batch_size=batch_size,
        num_runs=num_runs,
        warmup_queries=warmup,
        output_dir=output,
        validation_mode=ValidationMode(validation_mode),
        enable_tui=not no_tui,
        quiet=quiet,
        save_raw_outputs=save_raw,
        prompt_preset=prompt_preset,
        adaptive_prompt=adaptive_prompt,
    )

    asyncio.run(_run_benchmark(config))


@cli.command("quick-test")
@click.option(
    "--model",
    "-m",
    default="mlx-community/Qwen2-1.5B-Instruct-4bit",
    help="Model ID to evaluate",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    default="scripts/data",
    help="Path to dataset",
)
@click.option(
    "--sample-size",
    "-n",
    type=int,
    default=50,
    help="Number of queries to test",
)
def quick_test(model: str, dataset: str, sample_size: int):
    """Run a quick test with minimal queries."""
    config = BenchmarkConfig(
        dataset_path=dataset,
        model_id=model,
        sample_size=sample_size,
        num_runs=1,
        warmup_queries=5,
        batch_size=4,
        enable_tui=False,
        quiet=False,
    )

    asyncio.run(_run_benchmark(config))


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["all", "json", "markdown", "html", "csv"]),
    default="all",
    help="Report format to generate",
)
def report(results_dir: str, format: str):
    """Generate reports from existing results."""
    import json

    results_path = Path(results_dir)

    # Load existing metrics
    metrics_file = results_path / "metrics.json"
    config_file = results_path / "config.json"

    if not metrics_file.exists():
        logger.error("metrics.json not found in results directory")
        return

    with open(metrics_file) as f:
        metrics_data = json.load(f)

    # Reconstruct metrics snapshot
    from .metrics import MetricsSnapshot

    metrics = MetricsSnapshot(
        total_queries=metrics_data.get("total_queries", 0),
        valid_outputs=metrics_data.get("valid_outputs", 0),
        parse_errors=metrics_data.get("parse_errors", 0),
        medical_accuracy=metrics_data.get("accuracy", {}).get("medical", 0),
        medical_precision=metrics_data.get("accuracy", {}).get("medical_precision", 0),
        medical_recall=metrics_data.get("accuracy", {}).get("medical_recall", 0),
        medical_f1=metrics_data.get("accuracy", {}).get("medical_f1", 0),
        intent_accuracy=metrics_data.get("accuracy", {}).get("intent", 0),
        intent_per_class=metrics_data.get("accuracy", {}).get("intent_per_class", {}),
        entity_precision=metrics_data.get("extraction", {}).get("entity_precision", 0),
        entity_recall=metrics_data.get("extraction", {}).get("entity_recall", 0),
        entity_f1=metrics_data.get("extraction", {}).get("entity_f1", 0),
        relationship_precision=metrics_data.get("extraction", {}).get(
            "relationship_precision", 0
        ),
        relationship_recall=metrics_data.get("extraction", {}).get(
            "relationship_recall", 0
        ),
        relationship_f1=metrics_data.get("extraction", {}).get("relationship_f1", 0),
        latency_mean=metrics_data.get("performance", {}).get("latency_mean_ms", 0),
        latency_std=metrics_data.get("performance", {}).get("latency_std_ms", 0),
        latency_p50=metrics_data.get("performance", {}).get("latency_p50_ms", 0),
        latency_p90=metrics_data.get("performance", {}).get("latency_p90_ms", 0),
        latency_p95=metrics_data.get("performance", {}).get("latency_p95_ms", 0),
        latency_p99=metrics_data.get("performance", {}).get("latency_p99_ms", 0),
        throughput_qps=metrics_data.get("performance", {}).get("throughput_qps", 0),
        ece=metrics_data.get("calibration", {}).get("ece", 0),
        confidence_correlation=metrics_data.get("calibration", {}).get(
            "confidence_correlation", 0
        ),
        overconfidence_rate=metrics_data.get("calibration", {}).get(
            "overconfidence_rate", 0
        ),
        failure_modes=metrics_data.get("failure_modes", {}),
    )

    # Load config if available
    if config_file.exists():
        with open(config_file) as f:
            config_data = json.load(f)
        config = BenchmarkConfig(
            dataset_path=config_data.get("dataset_path", "scripts/data"),
            model_id=config_data.get("model_id", "unknown"),
            backend=config_data.get("backend", "mlx"),
        )
    else:
        config = BenchmarkConfig(
            dataset_path="scripts/data",
            model_id="unknown",
        )

    # Generate reports
    generator = ReportGenerator(config, metrics)

    if format == "all":
        paths = generator.generate_all(results_path)
        logger.info("Generated reports:")
        for fmt, path in paths.items():
            logger.info(f"  {fmt}: {path}")
    else:
        method = getattr(generator, f"generate_{format}")
        suffix = {"json": ".json", "markdown": ".md", "html": ".html", "csv": ".csv"}[
            format
        ]
        path = method(results_path / f"report{suffix}")
        logger.info(f"Generated {format} report: {path}")


@cli.command()
def list_models():
    """List available pre-configured models."""
    logger.info("MLX Models (Apple Silicon):")
    for name, config in MLX_MODELS.items():
        logger.info(f"  {name}: {config.model_id}")

    logger.info("Usage:")
    logger.info("  medquery-eval benchmark --model mlx-community/Qwen2-1.5B-Instruct-4bit")
    logger.info("  medquery-eval benchmark -m mlx-community/Phi-3-mini-4k-instruct-4bit")


async def _run_benchmark(config: BenchmarkConfig):
    """Run benchmark with TUI or quiet mode."""
    tui = None

    if config.enable_tui and not config.quiet:
        tui = BenchmarkTUI()
        runner = BenchmarkRunner(config, progress_callback=tui.progress_callback)
        tui.start()
    else:
        def log_progress(progress):
            if not config.quiet:
                # Use logger for proper verbose output
                if progress.latency_ms is not None:
                    valid_str = "valid" if progress.is_valid else "invalid"
                    logger.info(
                        f"{progress.phase.upper()} {progress.current}/{progress.total} - "
                        f"{valid_str}, {progress.latency_ms:.0f}ms"
                    )
                elif progress.current % 5 == 0 or progress.current == progress.total:
                    # Log every 5th warmup query or completion
                    logger.info(f"{progress.phase.upper()} {progress.current}/{progress.total}")

        runner = BenchmarkRunner(config, progress_callback=log_progress)

    try:
        metrics = await runner.run()

        if tui:
            tui.set_final_metrics(metrics)
            tui.stop()

        # Print final report
        if not config.quiet:
            print_final_report(metrics, console)

        # Generate reports
        generator = ReportGenerator(config, metrics, runner.get_collector())
        paths = generator.generate_all(config.results_dir)

        logger.info(f"Results saved to: {config.results_dir}")
        for fmt, path in paths.items():
            logger.info(f"  {fmt}: {path.name}")

    except Exception as e:
        if tui:
            tui.stop()
        logger.error(f"Benchmark error: {e}")
        raise click.Abort()


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
