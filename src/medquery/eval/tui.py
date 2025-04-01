"""Terminal UI dashboard for benchmark monitoring.

Real-time display of:
- Progress bars for warmup and evaluation
- Live metrics (accuracy, latency)
- Recent query results
- Error log
"""

from dataclasses import dataclass, field
from typing import Any
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.layout import Layout
from rich.text import Text

from .runner import RunProgress
from .metrics import MetricsSnapshot


@dataclass
class TUIState:
    """State for TUI dashboard."""

    phase: str = "initializing"
    current: int = 0
    total: int = 0

    # Running metrics
    valid_count: int = 0
    invalid_count: int = 0
    total_latency_ms: float = 0.0

    # Recent results (last 10)
    recent_results: list[dict] = field(default_factory=list)
    max_recent: int = 8

    # Errors
    errors: list[str] = field(default_factory=list)
    max_errors: int = 5

    # Final metrics
    final_metrics: MetricsSnapshot | None = None

    def update_from_progress(self, progress: RunProgress) -> None:
        """Update state from progress callback."""
        self.phase = progress.phase
        self.current = progress.current
        self.total = progress.total

        if progress.is_valid is not None:
            if progress.is_valid:
                self.valid_count += 1
            else:
                self.invalid_count += 1

        if progress.latency_ms is not None:
            self.total_latency_ms += progress.latency_ms

        if progress.error:
            self.errors.append(f"[{self.current}] {progress.error[:80]}")
            if len(self.errors) > self.max_errors:
                self.errors.pop(0)

        if progress.current_query and progress.phase == "evaluation":
            result = {
                "idx": self.current,
                "query": progress.current_query[:40] + "..." if len(progress.current_query) > 40 else progress.current_query,
                "valid": progress.is_valid,
                "latency": progress.latency_ms,
            }
            self.recent_results.append(result)
            if len(self.recent_results) > self.max_recent:
                self.recent_results.pop(0)

    @property
    def accuracy(self) -> float:
        """Current accuracy percentage."""
        total = self.valid_count + self.invalid_count
        if total == 0:
            return 0.0
        return (self.valid_count / total) * 100

    @property
    def avg_latency(self) -> float:
        """Average latency in ms."""
        total = self.valid_count + self.invalid_count
        if total == 0:
            return 0.0
        return self.total_latency_ms / total


class BenchmarkTUI:
    """Terminal UI for benchmark monitoring."""

    def __init__(self):
        self.console = Console()
        self.state = TUIState()
        self._live: Live | None = None
        self._start_time: float = 0.0

    def start(self) -> None:
        """Start the TUI display."""
        self._start_time = time.time()
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the TUI display."""
        if self._live:
            self._live.stop()

    def update(self, progress: RunProgress) -> None:
        """Update TUI with new progress."""
        self.state.update_from_progress(progress)
        if self._live:
            self._live.update(self._build_display())

    def set_final_metrics(self, metrics: MetricsSnapshot) -> None:
        """Set final metrics for display."""
        self.state.final_metrics = metrics

    def progress_callback(self, progress: RunProgress) -> None:
        """Callback for BenchmarkRunner."""
        self.update(progress)

    def _build_display(self) -> Panel:
        """Build the complete display layout."""
        layout = Layout()

        # Header
        header = self._build_header()

        # Progress section
        progress_panel = self._build_progress()

        # Metrics section
        metrics_panel = self._build_metrics()

        # Recent results
        results_panel = self._build_recent_results()

        # Errors
        errors_panel = self._build_errors()

        # Combine
        content = Group(
            header,
            "",
            progress_panel,
            "",
            metrics_panel,
            "",
            results_panel,
            "",
            errors_panel if self.state.errors else Text(""),
        )

        return Panel(
            content,
            title="[bold blue]MedQuery Benchmark[/bold blue]",
            border_style="blue",
        )

    def _build_header(self) -> Text:
        """Build header with phase info."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        elapsed_str = f"{elapsed:.1f}s"

        phase_colors = {
            "initializing": "yellow",
            "warmup": "cyan",
            "evaluation": "green",
            "computing": "magenta",
        }
        phase_color = phase_colors.get(self.state.phase, "white")

        text = Text()
        text.append("Phase: ", style="bold")
        text.append(self.state.phase.upper(), style=f"bold {phase_color}")
        text.append(f"  |  Elapsed: {elapsed_str}", style="dim")

        return text

    def _build_progress(self) -> Panel:
        """Build progress bar section."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
        )

        if self.state.total > 0:
            task = progress.add_task(
                f"[cyan]{self.state.phase.capitalize()}",
                total=self.state.total,
                completed=self.state.current,
            )
        else:
            task = progress.add_task("[yellow]Initializing...", total=100, completed=0)

        return Panel(progress, title="Progress", border_style="cyan")

    def _build_metrics(self) -> Panel:
        """Build live metrics panel."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        valid = self.state.valid_count
        invalid = self.state.invalid_count
        total = valid + invalid

        # Color accuracy based on value
        acc = self.state.accuracy
        if acc >= 90:
            acc_style = "bold green"
        elif acc >= 70:
            acc_style = "bold yellow"
        else:
            acc_style = "bold red"

        table.add_row(
            "Valid Outputs",
            f"[green]{valid}[/green]",
            "Parse Errors",
            f"[red]{invalid}[/red]",
        )
        table.add_row(
            "Parse Rate",
            f"[{acc_style}]{acc:.1f}%[/{acc_style}]",
            "Avg Latency",
            f"[cyan]{self.state.avg_latency:.0f}ms[/cyan]",
        )

        if total > 0:
            throughput = total / (time.time() - self._start_time) if self._start_time else 0
            table.add_row(
                "Throughput",
                f"[magenta]{throughput:.1f} q/s[/magenta]",
                "Processed",
                f"{total}",
            )

        return Panel(table, title="Live Metrics", border_style="green")

    def _build_recent_results(self) -> Panel:
        """Build recent results table."""
        table = Table(box=None, padding=(0, 1))
        table.add_column("#", style="dim", width=5)
        table.add_column("Query", width=45)
        table.add_column("Valid", width=6, justify="center")
        table.add_column("Latency", width=10, justify="right")

        for result in self.state.recent_results[-6:]:
            valid_str = "[green]\u2713[/green]" if result["valid"] else "[red]\u2717[/red]"
            latency_str = f"{result['latency']:.0f}ms" if result["latency"] else "-"
            table.add_row(
                str(result["idx"]),
                result["query"],
                valid_str,
                latency_str,
            )

        return Panel(table, title="Recent Results", border_style="yellow")

    def _build_errors(self) -> Panel:
        """Build errors panel."""
        if not self.state.errors:
            return Panel(Text("No errors", style="dim"), title="Errors", border_style="red")

        error_text = Text()
        for error in self.state.errors[-4:]:
            error_text.append(f"\u2022 {error}\n", style="red")

        return Panel(error_text, title=f"Errors ({len(self.state.errors)})", border_style="red")


def print_final_report(metrics: MetricsSnapshot, console: Console | None = None) -> None:
    """Print final metrics report to console."""
    console = console or Console()

    # Header
    console.print("\n")
    console.rule("[bold blue]Benchmark Results[/bold blue]")
    console.print("\n")

    # Summary table
    summary = Table(title="Summary", box=None)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")

    summary.add_row("Total Queries", str(metrics.total_queries))
    summary.add_row("Valid Outputs", f"[green]{metrics.valid_outputs}[/green]")
    summary.add_row("Parse Errors", f"[red]{metrics.parse_errors}[/red]")
    summary.add_row(
        "Parse Rate",
        f"{(metrics.valid_outputs / metrics.total_queries * 100):.1f}%" if metrics.total_queries > 0 else "N/A",
    )

    console.print(summary)
    console.print("\n")

    # Accuracy table
    acc_table = Table(title="Classification Accuracy", box=None)
    acc_table.add_column("Metric", style="bold")
    acc_table.add_column("Value", justify="right")

    acc_style = "green" if metrics.medical_accuracy >= 0.9 else "yellow" if metrics.medical_accuracy >= 0.7 else "red"
    acc_table.add_row("Medical Classification", f"[{acc_style}]{metrics.medical_accuracy*100:.1f}%[/{acc_style}]")
    acc_table.add_row("Medical F1", f"{metrics.medical_f1*100:.1f}%")

    intent_style = "green" if metrics.intent_accuracy >= 0.9 else "yellow" if metrics.intent_accuracy >= 0.7 else "red"
    acc_table.add_row("Intent Classification", f"[{intent_style}]{metrics.intent_accuracy*100:.1f}%[/{intent_style}]")

    # Per-class intent
    for intent, acc in metrics.intent_per_class.items():
        acc_table.add_row(f"  {intent}", f"{acc*100:.1f}%")

    console.print(acc_table)
    console.print("\n")

    # Extraction metrics
    if metrics.entity_f1 > 0 or metrics.relationship_f1 > 0:
        ext_table = Table(title="Extraction Metrics", box=None)
        ext_table.add_column("Metric", style="bold")
        ext_table.add_column("Precision", justify="right")
        ext_table.add_column("Recall", justify="right")
        ext_table.add_column("F1", justify="right")

        ext_table.add_row(
            "Entities",
            f"{metrics.entity_precision*100:.1f}%",
            f"{metrics.entity_recall*100:.1f}%",
            f"{metrics.entity_f1*100:.1f}%",
        )
        ext_table.add_row(
            "Relationships",
            f"{metrics.relationship_precision*100:.1f}%",
            f"{metrics.relationship_recall*100:.1f}%",
            f"{metrics.relationship_f1*100:.1f}%",
        )

        console.print(ext_table)
        console.print("\n")

    # Performance table
    perf_table = Table(title="Performance", box=None)
    perf_table.add_column("Metric", style="bold")
    perf_table.add_column("Value", justify="right")

    perf_table.add_row("Mean Latency", f"{metrics.latency_mean:.0f}ms")
    perf_table.add_row("P50 Latency", f"{metrics.latency_p50:.0f}ms")
    perf_table.add_row("P90 Latency", f"{metrics.latency_p90:.0f}ms")
    perf_table.add_row("P99 Latency", f"{metrics.latency_p99:.0f}ms")
    perf_table.add_row("Throughput", f"{metrics.throughput_qps:.2f} q/s")

    console.print(perf_table)
    console.print("\n")

    # Failure modes
    if metrics.failure_modes:
        fail_table = Table(title="Failure Analysis", box=None)
        fail_table.add_column("Mode", style="bold")
        fail_table.add_column("Count", justify="right")

        for mode, count in sorted(metrics.failure_modes.items(), key=lambda x: -x[1]):
            fail_table.add_row(mode, str(count))

        console.print(fail_table)

    console.print("\n")
    console.rule("[bold blue]End of Report[/bold blue]")
