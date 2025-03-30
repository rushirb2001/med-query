"""Benchmark configuration for evaluation runs.

Defines configuration dataclasses and preset configurations
for quick testing, standard benchmarks, and thorough evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from datetime import datetime

from .validators import ValidationMode


BackendType = Literal["mlx", "llamacpp", "vllm", "openai", "anthropic"]


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    # Dataset
    dataset_path: str | Path
    split: Literal["train", "val", "test"] = "test"
    sample_size: int | None = None  # None = full dataset
    categories: list[str] | None = None  # None = all categories

    # Execution
    batch_size: int = 8
    warmup_queries: int = 10
    num_runs: int = 3
    seed: int = 42

    # Model
    backend: BackendType = "mlx"
    model_id: str = "mlx-community/Qwen2-1.5B-Instruct-4bit"
    max_tokens: int = 256
    temperature: float = 0.0  # Deterministic for benchmarks

    # Validation
    validation_mode: ValidationMode = ValidationMode.LENIENT

    # Output
    output_dir: str | Path = "results"
    run_name: str | None = None  # Auto-generated if None
    save_predictions: bool = True
    save_raw_outputs: bool = False

    # TUI
    enable_tui: bool = True
    quiet: bool = False

    def __post_init__(self):
        """Generate run name if not provided."""
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short = self.model_id.split("/")[-1][:20]
            self.run_name = f"{timestamp}_{model_short}"

        # Convert paths
        self.dataset_path = Path(self.dataset_path)
        self.output_dir = Path(self.output_dir)

    @property
    def results_dir(self) -> Path:
        """Get full results directory path."""
        return self.output_dir / self.run_name

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "dataset_path": str(self.dataset_path),
            "split": self.split,
            "sample_size": self.sample_size,
            "categories": self.categories,
            "batch_size": self.batch_size,
            "warmup_queries": self.warmup_queries,
            "num_runs": self.num_runs,
            "seed": self.seed,
            "backend": self.backend,
            "model_id": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "validation_mode": self.validation_mode.value,
            "output_dir": str(self.output_dir),
            "run_name": self.run_name,
            "save_predictions": self.save_predictions,
            "save_raw_outputs": self.save_raw_outputs,
            "enable_tui": self.enable_tui,
            "quiet": self.quiet,
        }


# Preset configurations
QUICK_TEST = BenchmarkConfig(
    dataset_path="scripts/data",
    sample_size=100,
    num_runs=1,
    warmup_queries=5,
    batch_size=4,
    enable_tui=False,
)

STANDARD = BenchmarkConfig(
    dataset_path="scripts/data",
    num_runs=3,
    warmup_queries=10,
    batch_size=8,
)

THOROUGH = BenchmarkConfig(
    dataset_path="scripts/data",
    num_runs=5,
    warmup_queries=20,
    batch_size=1,  # No batching for accurate per-query latency
)

LATENCY_FOCUSED = BenchmarkConfig(
    dataset_path="scripts/data",
    sample_size=500,
    num_runs=5,
    warmup_queries=50,
    batch_size=1,
)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    backend: BackendType
    display_name: str | None = None
    system_prompt: str | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    extra_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = self.model_id.split("/")[-1]


# Preset model configurations
MLX_MODELS = {
    "qwen2-1.5b": ModelConfig(
        model_id="mlx-community/Qwen2-1.5B-Instruct-4bit",
        backend="mlx",
        display_name="Qwen2-1.5B (4-bit)",
    ),
    "qwen2-0.5b": ModelConfig(
        model_id="mlx-community/Qwen2-0.5B-Instruct-4bit",
        backend="mlx",
        display_name="Qwen2-0.5B (4-bit)",
    ),
    "phi3-mini": ModelConfig(
        model_id="mlx-community/Phi-3-mini-4k-instruct-4bit",
        backend="mlx",
        display_name="Phi-3-mini (4-bit)",
    ),
    "gemma2-2b": ModelConfig(
        model_id="mlx-community/gemma-2-2b-it-4bit",
        backend="mlx",
        display_name="Gemma-2-2B (4-bit)",
    ),
}

API_MODELS = {
    "claude-haiku": ModelConfig(
        model_id="claude-3-haiku-20240307",
        backend="anthropic",
        display_name="Claude 3 Haiku",
    ),
    "gpt4-mini": ModelConfig(
        model_id="gpt-4o-mini",
        backend="openai",
        display_name="GPT-4o Mini",
    ),
}


def load_config_from_file(path: str | Path) -> BenchmarkConfig:
    """Load configuration from YAML or JSON file.

    Args:
        path: Path to config file

    Returns:
        BenchmarkConfig instance
    """
    import json
    import yaml

    path = Path(path)

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    # Convert validation_mode string to enum
    if "validation_mode" in data:
        data["validation_mode"] = ValidationMode(data["validation_mode"])

    return BenchmarkConfig(**data)


def save_config_to_file(config: BenchmarkConfig, path: str | Path) -> None:
    """Save configuration to YAML or JSON file.

    Args:
        config: BenchmarkConfig instance
        path: Output path
    """
    import json
    import yaml

    path = Path(path)
    data = config.to_dict()

    with open(path, "w") as f:
        if path.suffix in (".yaml", ".yml"):
            yaml.dump(data, f, default_flow_style=False)
        else:
            json.dump(data, f, indent=2)
