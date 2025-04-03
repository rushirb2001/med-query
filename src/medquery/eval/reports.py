"""Report generation for benchmark results.

Generates reports in multiple formats:
- JSON: Machine-readable full data
- Markdown: Human-readable summary
- HTML: Interactive dashboard
- CSV: Spreadsheet-compatible data
"""

import json
from datetime import datetime
from pathlib import Path

from .metrics import MetricsSnapshot, MetricsCollector
from .config import BenchmarkConfig
from ..logging import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """Generate benchmark reports in multiple formats."""

    def __init__(
        self,
        config: BenchmarkConfig,
        metrics: MetricsSnapshot,
        collector: MetricsCollector | None = None,
    ):
        """Initialize report generator.

        Args:
            config: Benchmark configuration
            metrics: Aggregated metrics
            collector: Optional collector for detailed results
        """
        self.config = config
        self.metrics = metrics
        self.collector = collector
        self.timestamp = datetime.now().isoformat()

    def generate_all(self, output_dir: Path | str) -> dict[str, Path]:
        """Generate all report formats.

        Args:
            output_dir: Directory to save reports

        Returns:
            Dictionary mapping format to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating reports in {output_dir}")

        paths = {}
        paths["json"] = self.generate_json(output_dir / "report.json")
        paths["markdown"] = self.generate_markdown(output_dir / "report.md")
        paths["html"] = self.generate_html(output_dir / "report.html")
        paths["csv"] = self.generate_csv(output_dir / "results.csv")

        logger.debug(f"Generated {len(paths)} report formats")
        return paths

    def generate_json(self, path: Path) -> Path:
        """Generate JSON report."""
        report = {
            "meta": {
                "timestamp": self.timestamp,
                "config": self.config.to_dict(),
            },
            "summary": {
                "total_queries": self.metrics.total_queries,
                "valid_outputs": self.metrics.valid_outputs,
                "parse_errors": self.metrics.parse_errors,
                "parse_rate": self.metrics.valid_outputs / self.metrics.total_queries
                if self.metrics.total_queries > 0
                else 0,
            },
            "accuracy": {
                "medical": self.metrics.medical_accuracy,
                "medical_precision": self.metrics.medical_precision,
                "medical_recall": self.metrics.medical_recall,
                "medical_f1": self.metrics.medical_f1,
                "intent": self.metrics.intent_accuracy,
                "intent_per_class": self.metrics.intent_per_class,
            },
            "extraction": {
                "entity_precision": self.metrics.entity_precision,
                "entity_recall": self.metrics.entity_recall,
                "entity_f1": self.metrics.entity_f1,
                "relationship_precision": self.metrics.relationship_precision,
                "relationship_recall": self.metrics.relationship_recall,
                "relationship_f1": self.metrics.relationship_f1,
            },
            "performance": {
                "latency_mean_ms": self.metrics.latency_mean,
                "latency_std_ms": self.metrics.latency_std,
                "latency_p50_ms": self.metrics.latency_p50,
                "latency_p90_ms": self.metrics.latency_p90,
                "latency_p95_ms": self.metrics.latency_p95,
                "latency_p99_ms": self.metrics.latency_p99,
                "throughput_qps": self.metrics.throughput_qps,
            },
            "calibration": {
                "ece": self.metrics.ece,
                "confidence_correlation": self.metrics.confidence_correlation,
                "overconfidence_rate": self.metrics.overconfidence_rate,
            },
            "failure_modes": self.metrics.failure_modes,
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        return path

    def generate_markdown(self, path: Path) -> Path:
        """Generate Markdown report."""
        m = self.metrics

        md = f"""# MedQuery Benchmark Report

**Generated:** {self.timestamp}
**Model:** {self.config.model_id}
**Backend:** {self.config.backend}

## Summary

| Metric | Value |
|--------|-------|
| Total Queries | {m.total_queries} |
| Valid Outputs | {m.valid_outputs} |
| Parse Errors | {m.parse_errors} |
| Parse Rate | {m.valid_outputs / m.total_queries * 100:.1f}% |

## Classification Accuracy

### Medical Classification

| Metric | Value |
|--------|-------|
| Accuracy | {m.medical_accuracy * 100:.1f}% |
| Precision | {m.medical_precision * 100:.1f}% |
| Recall | {m.medical_recall * 100:.1f}% |
| F1 Score | {m.medical_f1 * 100:.1f}% |

### Intent Classification

| Metric | Value |
|--------|-------|
| Overall Accuracy | {m.intent_accuracy * 100:.1f}% |
"""

        if m.intent_per_class:
            for intent, acc in m.intent_per_class.items():
                md += f"| {intent.capitalize()} | {acc * 100:.1f}% |\n"

        md += f"""
## Extraction Metrics

### Entity Extraction

| Metric | Value |
|--------|-------|
| Precision | {m.entity_precision * 100:.1f}% |
| Recall | {m.entity_recall * 100:.1f}% |
| F1 Score | {m.entity_f1 * 100:.1f}% |

### Relationship Extraction

| Metric | Value |
|--------|-------|
| Precision | {m.relationship_precision * 100:.1f}% |
| Recall | {m.relationship_recall * 100:.1f}% |
| F1 Score | {m.relationship_f1 * 100:.1f}% |

## Performance

| Metric | Value |
|--------|-------|
| Mean Latency | {m.latency_mean:.0f}ms |
| Std Latency | {m.latency_std:.0f}ms |
| P50 Latency | {m.latency_p50:.0f}ms |
| P90 Latency | {m.latency_p90:.0f}ms |
| P95 Latency | {m.latency_p95:.0f}ms |
| P99 Latency | {m.latency_p99:.0f}ms |
| Throughput | {m.throughput_qps:.2f} queries/sec |

## Calibration

| Metric | Value |
|--------|-------|
| Expected Calibration Error | {m.ece:.4f} |
| Confidence Correlation | {m.confidence_correlation:.4f} |
| Overconfidence Rate | {m.overconfidence_rate * 100:.1f}% |

## Failure Analysis

"""

        if m.failure_modes:
            md += "| Mode | Count |\n|------|-------|\n"
            for mode, count in sorted(m.failure_modes.items(), key=lambda x: -x[1]):
                md += f"| {mode} | {count} |\n"
        else:
            md += "No failures recorded.\n"

        md += """
---
*Report generated by MedQuery Benchmark Framework*
"""

        with open(path, "w") as f:
            f.write(md)

        return path

    def generate_html(self, path: Path) -> Path:
        """Generate HTML report with embedded styles."""
        m = self.metrics

        # Determine color classes
        def acc_class(val: float) -> str:
            if val >= 0.9:
                return "good"
            elif val >= 0.7:
                return "warn"
            return "bad"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedQuery Benchmark Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
        h2 {{ color: #1e40af; margin-top: 30px; }}
        .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card h3 {{ margin-top: 0; color: #374151; border-bottom: 1px solid #e5e7eb; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f9fafb; font-weight: 600; }}
        .value {{ font-weight: 600; font-size: 1.1em; }}
        .good {{ color: #059669; }}
        .warn {{ color: #d97706; }}
        .bad {{ color: #dc2626; }}
        .metric-large {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{ color: #6b7280; font-size: 0.9em; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
        .summary-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .bar-container {{ background: #e5e7eb; border-radius: 4px; height: 8px; margin-top: 5px; }}
        .bar {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
        .bar.good {{ background: #059669; }}
        .bar.warn {{ background: #d97706; }}
        .bar.bad {{ background: #dc2626; }}
    </style>
</head>
<body>
    <h1>MedQuery Benchmark Report</h1>
    <div class="meta">
        <strong>Model:</strong> {self.config.model_id}<br>
        <strong>Backend:</strong> {self.config.backend}<br>
        <strong>Generated:</strong> {self.timestamp}
    </div>

    <div class="summary-grid">
        <div class="summary-item">
            <div class="metric-label">Total Queries</div>
            <div class="metric-large">{m.total_queries}</div>
        </div>
        <div class="summary-item">
            <div class="metric-label">Valid Outputs</div>
            <div class="metric-large good">{m.valid_outputs}</div>
        </div>
        <div class="summary-item">
            <div class="metric-label">Parse Rate</div>
            <div class="metric-large {acc_class(m.valid_outputs/m.total_queries if m.total_queries > 0 else 0)}">{m.valid_outputs/m.total_queries*100:.1f}%</div>
        </div>
        <div class="summary-item">
            <div class="metric-label">Throughput</div>
            <div class="metric-large">{m.throughput_qps:.1f} q/s</div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>Medical Classification</h3>
            <table>
                <tr>
                    <td>Accuracy</td>
                    <td class="value {acc_class(m.medical_accuracy)}">{m.medical_accuracy*100:.1f}%</td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td class="value">{m.medical_precision*100:.1f}%</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td class="value">{m.medical_recall*100:.1f}%</td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td class="value {acc_class(m.medical_f1)}">{m.medical_f1*100:.1f}%</td>
                </tr>
            </table>
            <div class="bar-container">
                <div class="bar {acc_class(m.medical_accuracy)}" style="width: {m.medical_accuracy*100}%"></div>
            </div>
        </div>

        <div class="card">
            <h3>Intent Classification</h3>
            <table>
                <tr>
                    <td>Overall Accuracy</td>
                    <td class="value {acc_class(m.intent_accuracy)}">{m.intent_accuracy*100:.1f}%</td>
                </tr>
"""

        for intent, acc in m.intent_per_class.items():
            html += f"""                <tr>
                    <td>{intent.capitalize()}</td>
                    <td class="value">{acc*100:.1f}%</td>
                </tr>
"""

        html += f"""            </table>
            <div class="bar-container">
                <div class="bar {acc_class(m.intent_accuracy)}" style="width: {m.intent_accuracy*100}%"></div>
            </div>
        </div>

        <div class="card">
            <h3>Entity Extraction</h3>
            <table>
                <tr>
                    <td>Precision</td>
                    <td class="value">{m.entity_precision*100:.1f}%</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td class="value">{m.entity_recall*100:.1f}%</td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td class="value {acc_class(m.entity_f1)}">{m.entity_f1*100:.1f}%</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h3>Relationship Extraction</h3>
            <table>
                <tr>
                    <td>Precision</td>
                    <td class="value">{m.relationship_precision*100:.1f}%</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td class="value">{m.relationship_recall*100:.1f}%</td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td class="value {acc_class(m.relationship_f1)}">{m.relationship_f1*100:.1f}%</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h3>Performance</h3>
            <table>
                <tr>
                    <td>Mean Latency</td>
                    <td class="value">{m.latency_mean:.0f}ms</td>
                </tr>
                <tr>
                    <td>P50 Latency</td>
                    <td class="value">{m.latency_p50:.0f}ms</td>
                </tr>
                <tr>
                    <td>P90 Latency</td>
                    <td class="value">{m.latency_p90:.0f}ms</td>
                </tr>
                <tr>
                    <td>P99 Latency</td>
                    <td class="value">{m.latency_p99:.0f}ms</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h3>Calibration</h3>
            <table>
                <tr>
                    <td>Expected Calibration Error</td>
                    <td class="value">{m.ece:.4f}</td>
                </tr>
                <tr>
                    <td>Confidence Correlation</td>
                    <td class="value">{m.confidence_correlation:.4f}</td>
                </tr>
                <tr>
                    <td>Overconfidence Rate</td>
                    <td class="value {acc_class(1-m.overconfidence_rate)}">{m.overconfidence_rate*100:.1f}%</td>
                </tr>
            </table>
        </div>
    </div>

    <h2>Failure Analysis</h2>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Failure Mode</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""

        if m.failure_modes:
            for mode, count in sorted(m.failure_modes.items(), key=lambda x: -x[1]):
                pct = count / m.total_queries * 100 if m.total_queries > 0 else 0
                html += f"""                <tr>
                    <td>{mode}</td>
                    <td>{count}</td>
                    <td>{pct:.1f}%</td>
                </tr>
"""
        else:
            html += """                <tr>
                    <td colspan="3" style="text-align: center; color: #059669;">No failures recorded</td>
                </tr>
"""

        html += """            </tbody>
        </table>
    </div>

    <footer style="margin-top: 40px; text-align: center; color: #6b7280; font-size: 0.9em;">
        <p>Generated by MedQuery Benchmark Framework</p>
    </footer>
</body>
</html>
"""

        with open(path, "w") as f:
            f.write(html)

        return path

    def generate_csv(self, path: Path) -> Path:
        """Generate CSV with per-query results."""
        if not self.collector:
            # Just write metrics summary
            lines = [
                "metric,value",
                f"total_queries,{self.metrics.total_queries}",
                f"valid_outputs,{self.metrics.valid_outputs}",
                f"medical_accuracy,{self.metrics.medical_accuracy:.4f}",
                f"intent_accuracy,{self.metrics.intent_accuracy:.4f}",
                f"entity_f1,{self.metrics.entity_f1:.4f}",
                f"relationship_f1,{self.metrics.relationship_f1:.4f}",
                f"latency_mean_ms,{self.metrics.latency_mean:.2f}",
                f"latency_p50_ms,{self.metrics.latency_p50:.2f}",
                f"latency_p90_ms,{self.metrics.latency_p90:.2f}",
                f"throughput_qps,{self.metrics.throughput_qps:.4f}",
            ]
            with open(path, "w") as f:
                f.write("\n".join(lines))
            return path

        # Full per-query results
        lines = [
            "query_id,category,level,is_valid,medical_correct,intent_correct,latency_ms"
        ]

        for r in self.collector.results:
            medical_correct = 1 if r.medical_correct else 0 if r.medical_correct is False else ""
            intent_correct = 1 if r.intent_correct else 0 if r.intent_correct is False else ""
            lines.append(
                f"{r.query_id},{r.category},{r.level},{int(r.is_valid)},"
                f"{medical_correct},{intent_correct},{r.latency_ms:.2f}"
            )

        with open(path, "w") as f:
            f.write("\n".join(lines))

        return path
