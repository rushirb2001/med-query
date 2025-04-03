"""Metrics collection and aggregation for benchmark evaluation.

Tracks:
- Classification accuracy (medical, intent)
- Entity/relationship extraction (precision, recall, F1)
- Performance metrics (latency percentiles, throughput)
- Calibration metrics (ECE, confidence correlation)
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .schemas import QueryAnalysis, ExpectedOutput, ValidationResult
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Result of evaluating a single query."""

    query_id: str
    query: str
    category: str
    level: str
    expected: ExpectedOutput
    predicted: QueryAnalysis | None
    validation: ValidationResult
    latency_ms: float
    tokens_generated: int = 0

    @property
    def is_valid(self) -> bool:
        return self.validation.valid and self.predicted is not None

    @property
    def medical_correct(self) -> bool | None:
        if not self.is_valid:
            return None
        return self.predicted.is_medical == self.expected.is_medical

    @property
    def intent_correct(self) -> bool | None:
        if not self.is_valid:
            return None
        if self.expected.primary_intent is None:
            return True
        return self.predicted.primary_intent == self.expected.primary_intent


@dataclass
class MetricsSnapshot:
    """Aggregated metrics from evaluation run."""

    # Counts
    total_queries: int = 0
    valid_outputs: int = 0
    parse_errors: int = 0

    # Classification accuracy
    medical_accuracy: float = 0.0
    medical_precision: float = 0.0
    medical_recall: float = 0.0
    medical_f1: float = 0.0

    intent_accuracy: float = 0.0
    intent_per_class: dict[str, float] = field(default_factory=dict)

    # Entity extraction
    entity_precision: float = 0.0
    entity_recall: float = 0.0
    entity_f1: float = 0.0

    # Relationship extraction
    relationship_precision: float = 0.0
    relationship_recall: float = 0.0
    relationship_f1: float = 0.0

    # Performance
    latency_mean: float = 0.0
    latency_std: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput_qps: float = 0.0
    tokens_per_second: float = 0.0

    # Calibration
    ece: float = 0.0  # Expected Calibration Error
    confidence_correlation: float = 0.0
    overconfidence_rate: float = 0.0

    # Breakdowns
    by_level: dict[str, "MetricsSnapshot"] = field(default_factory=dict)
    by_category: dict[str, "MetricsSnapshot"] = field(default_factory=dict)

    # Failure analysis
    failure_modes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_queries": self.total_queries,
            "valid_outputs": self.valid_outputs,
            "parse_errors": self.parse_errors,
            "accuracy": {
                "medical": self.medical_accuracy,
                "medical_precision": self.medical_precision,
                "medical_recall": self.medical_recall,
                "medical_f1": self.medical_f1,
                "intent": self.intent_accuracy,
                "intent_per_class": self.intent_per_class,
            },
            "extraction": {
                "entity_precision": self.entity_precision,
                "entity_recall": self.entity_recall,
                "entity_f1": self.entity_f1,
                "relationship_precision": self.relationship_precision,
                "relationship_recall": self.relationship_recall,
                "relationship_f1": self.relationship_f1,
            },
            "performance": {
                "latency_mean_ms": self.latency_mean,
                "latency_std_ms": self.latency_std,
                "latency_p50_ms": self.latency_p50,
                "latency_p90_ms": self.latency_p90,
                "latency_p95_ms": self.latency_p95,
                "latency_p99_ms": self.latency_p99,
                "throughput_qps": self.throughput_qps,
                "tokens_per_second": self.tokens_per_second,
            },
            "calibration": {
                "ece": self.ece,
                "confidence_correlation": self.confidence_correlation,
                "overconfidence_rate": self.overconfidence_rate,
            },
            "failure_modes": self.failure_modes,
        }


class MetricsCollector:
    """Collect and aggregate metrics from query results."""

    def __init__(self):
        self.results: list[QueryResult] = []
        self._total_time_ms: float = 0.0

    def add(self, result: QueryResult) -> None:
        """Add a query result."""
        self.results.append(result)
        self._total_time_ms += result.latency_ms
        logger.debug(f"Added result for query {result.query_id}: valid={result.is_valid}, latency={result.latency_ms:.1f}ms")

    def clear(self) -> None:
        """Clear all results."""
        self.results.clear()
        self._total_time_ms = 0.0

    def compute(self, include_breakdowns: bool = True) -> MetricsSnapshot:
        """Compute aggregated metrics from all results.

        Args:
            include_breakdowns: Whether to compute by_level and by_category breakdowns.
                               Set to False for nested computations to avoid recursion.
        """
        if not self.results:
            logger.warning("No results to compute metrics from")
            return MetricsSnapshot()

        logger.debug(f"Computing metrics from {len(self.results)} results")

        snapshot = MetricsSnapshot()
        snapshot.total_queries = len(self.results)

        # Count valid/invalid
        valid_results = [r for r in self.results if r.is_valid]
        snapshot.valid_outputs = len(valid_results)
        snapshot.parse_errors = snapshot.total_queries - snapshot.valid_outputs

        if not valid_results:
            return snapshot

        # Classification accuracy
        self._compute_medical_accuracy(valid_results, snapshot)
        self._compute_intent_accuracy(valid_results, snapshot)

        # Entity/relationship extraction
        self._compute_entity_metrics(valid_results, snapshot)
        self._compute_relationship_metrics(valid_results, snapshot)

        # Performance
        self._compute_latency_metrics(snapshot)

        # Calibration
        self._compute_calibration_metrics(valid_results, snapshot)

        # Failure analysis
        self._compute_failure_modes(snapshot)

        # Breakdowns (skip for nested to avoid recursion)
        if include_breakdowns:
            self._compute_breakdowns(snapshot)

        logger.info(f"Metrics computed: {snapshot.valid_outputs}/{snapshot.total_queries} valid, "
                   f"medical_acc={snapshot.medical_accuracy:.1%}, intent_acc={snapshot.intent_accuracy:.1%}")
        return snapshot

    def _compute_medical_accuracy(
        self, results: list[QueryResult], snapshot: MetricsSnapshot
    ) -> None:
        """Compute medical classification metrics."""
        correct = sum(1 for r in results if r.medical_correct)
        snapshot.medical_accuracy = correct / len(results)

        # Precision, recall, F1 for medical=True
        tp = sum(
            1
            for r in results
            if r.predicted.is_medical and r.expected.is_medical
        )
        fp = sum(
            1
            for r in results
            if r.predicted.is_medical and not r.expected.is_medical
        )
        fn = sum(
            1
            for r in results
            if not r.predicted.is_medical and r.expected.is_medical
        )

        snapshot.medical_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        snapshot.medical_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if snapshot.medical_precision + snapshot.medical_recall > 0:
            snapshot.medical_f1 = (
                2
                * snapshot.medical_precision
                * snapshot.medical_recall
                / (snapshot.medical_precision + snapshot.medical_recall)
            )

    def _compute_intent_accuracy(
        self, results: list[QueryResult], snapshot: MetricsSnapshot
    ) -> None:
        """Compute intent classification metrics."""
        # Filter to results with expected intent
        intent_results = [r for r in results if r.expected.primary_intent is not None]
        if not intent_results:
            snapshot.intent_accuracy = 1.0  # No intent to check
            return

        correct = sum(1 for r in intent_results if r.intent_correct)
        snapshot.intent_accuracy = correct / len(intent_results)

        # Per-class accuracy
        intent_classes = {"conceptual", "procedural", "relationship", "lookup"}
        for intent in intent_classes:
            class_results = [
                r for r in intent_results if r.expected.primary_intent == intent
            ]
            if class_results:
                class_correct = sum(1 for r in class_results if r.intent_correct)
                snapshot.intent_per_class[intent] = class_correct / len(class_results)

    def _compute_entity_metrics(
        self, results: list[QueryResult], snapshot: MetricsSnapshot
    ) -> None:
        """Compute entity extraction metrics."""
        precisions = []
        recalls = []

        for r in results:
            if not r.expected.entities:
                if not r.predicted.entities:
                    precisions.append(1.0)
                    recalls.append(1.0)
                continue

            if not r.predicted.entities:
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            pred_set = {(e.text.lower(), e.type) for e in r.predicted.entities}
            exp_set = {(e.text.lower(), e.type) for e in r.expected.entities}

            tp = len(pred_set & exp_set)
            precisions.append(tp / len(pred_set) if pred_set else 0.0)
            recalls.append(tp / len(exp_set) if exp_set else 0.0)

        if precisions:
            snapshot.entity_precision = np.mean(precisions)
            snapshot.entity_recall = np.mean(recalls)
            if snapshot.entity_precision + snapshot.entity_recall > 0:
                snapshot.entity_f1 = (
                    2
                    * snapshot.entity_precision
                    * snapshot.entity_recall
                    / (snapshot.entity_precision + snapshot.entity_recall)
                )

    def _compute_relationship_metrics(
        self, results: list[QueryResult], snapshot: MetricsSnapshot
    ) -> None:
        """Compute relationship extraction metrics."""
        precisions = []
        recalls = []

        for r in results:
            if not r.expected.relationships:
                if not r.predicted.relationships:
                    precisions.append(1.0)
                    recalls.append(1.0)
                continue

            if not r.predicted.relationships:
                precisions.append(0.0)
                recalls.append(0.0)
                continue

            pred_set = {
                (rel.source.lower(), rel.target.lower(), rel.type)
                for rel in r.predicted.relationships
            }
            exp_set = {
                (rel.source.lower(), rel.target.lower(), rel.type)
                for rel in r.expected.relationships
            }

            tp = len(pred_set & exp_set)
            precisions.append(tp / len(pred_set) if pred_set else 0.0)
            recalls.append(tp / len(exp_set) if exp_set else 0.0)

        if precisions:
            snapshot.relationship_precision = np.mean(precisions)
            snapshot.relationship_recall = np.mean(recalls)
            if snapshot.relationship_precision + snapshot.relationship_recall > 0:
                snapshot.relationship_f1 = (
                    2
                    * snapshot.relationship_precision
                    * snapshot.relationship_recall
                    / (snapshot.relationship_precision + snapshot.relationship_recall)
                )

    def _compute_latency_metrics(self, snapshot: MetricsSnapshot) -> None:
        """Compute latency percentiles and throughput."""
        latencies = [r.latency_ms for r in self.results]

        snapshot.latency_mean = np.mean(latencies)
        snapshot.latency_std = np.std(latencies)
        snapshot.latency_p50 = np.percentile(latencies, 50)
        snapshot.latency_p90 = np.percentile(latencies, 90)
        snapshot.latency_p95 = np.percentile(latencies, 95)
        snapshot.latency_p99 = np.percentile(latencies, 99)

        # Throughput
        if self._total_time_ms > 0:
            snapshot.throughput_qps = len(self.results) / (self._total_time_ms / 1000)

        # Tokens per second
        total_tokens = sum(r.tokens_generated for r in self.results)
        if self._total_time_ms > 0 and total_tokens > 0:
            snapshot.tokens_per_second = total_tokens / (self._total_time_ms / 1000)

    def _compute_calibration_metrics(
        self, results: list[QueryResult], snapshot: MetricsSnapshot
    ) -> None:
        """Compute calibration metrics (ECE, confidence correlation)."""
        confidences = [r.predicted.confidence for r in results]
        correct = [1.0 if r.medical_correct else 0.0 for r in results]

        if not confidences:
            return

        # Expected Calibration Error (ECE) - 10 bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_mask = [
                bin_boundaries[i] <= c < bin_boundaries[i + 1] for c in confidences
            ]
            bin_count = sum(bin_mask)
            if bin_count == 0:
                continue

            bin_conf = np.mean([c for c, m in zip(confidences, bin_mask) if m])
            bin_acc = np.mean([a for a, m in zip(correct, bin_mask) if m])
            ece += (bin_count / len(confidences)) * abs(bin_acc - bin_conf)

        snapshot.ece = ece

        # Confidence correlation (Pearson)
        # Suppress warning when std=0 (all values identical)
        if len(confidences) > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                snapshot.confidence_correlation = np.corrcoef(confidences, correct)[0, 1]
            if np.isnan(snapshot.confidence_correlation):
                snapshot.confidence_correlation = 0.0

        # Overconfidence rate (wrong predictions with confidence > 0.8)
        wrong_high_conf = sum(
            1
            for r in results
            if not r.medical_correct and r.predicted.confidence > 0.8
        )
        snapshot.overconfidence_rate = wrong_high_conf / len(results)

    def _compute_failure_modes(self, snapshot: MetricsSnapshot) -> None:
        """Analyze and categorize failure modes."""
        failure_modes: dict[str, int] = {}

        for r in self.results:
            if not r.is_valid:
                mode = "parse_error"
            elif not r.medical_correct:
                if r.predicted.is_medical and not r.expected.is_medical:
                    mode = "false_positive_medical"
                else:
                    mode = "false_negative_medical"
            elif not r.intent_correct:
                pred = r.predicted.primary_intent or "none"
                exp = r.expected.primary_intent or "none"
                mode = f"{exp}_as_{pred}"
            else:
                continue  # Not a failure

            failure_modes[mode] = failure_modes.get(mode, 0) + 1

        snapshot.failure_modes = failure_modes

    def _compute_breakdowns(self, snapshot: MetricsSnapshot) -> None:
        """Compute metrics broken down by level and category."""
        # By level
        levels = set(r.level for r in self.results)
        for level in levels:
            level_results = [r for r in self.results if r.level == level]
            collector = MetricsCollector()
            collector.results = level_results
            collector._total_time_ms = sum(r.latency_ms for r in level_results)
            # Pass include_breakdowns=False to avoid recursion
            snapshot.by_level[level] = collector.compute(include_breakdowns=False)

        # By category
        categories = set(r.category for r in self.results)
        for category in categories:
            cat_results = [r for r in self.results if r.category == category]
            collector = MetricsCollector()
            collector.results = cat_results
            collector._total_time_ms = sum(r.latency_ms for r in cat_results)
            # Pass include_breakdowns=False to avoid recursion
            snapshot.by_category[category] = collector.compute(include_breakdowns=False)

    def compute_confusion_matrix(
        self, field: str = "intent"
    ) -> tuple[np.ndarray, list[str]]:
        """Compute confusion matrix for a classification field.

        Args:
            field: "intent" or "medical"

        Returns:
            Tuple of (confusion_matrix, labels)
        """
        valid_results = [r for r in self.results if r.is_valid]

        if field == "medical":
            labels = ["non_medical", "medical"]
            matrix = np.zeros((2, 2), dtype=int)
            for r in valid_results:
                pred_idx = 1 if r.predicted.is_medical else 0
                exp_idx = 1 if r.expected.is_medical else 0
                matrix[exp_idx, pred_idx] += 1

        elif field == "intent":
            labels = ["conceptual", "procedural", "relationship", "lookup"]
            matrix = np.zeros((4, 4), dtype=int)
            label_to_idx = {l: i for i, l in enumerate(labels)}

            for r in valid_results:
                if r.expected.primary_intent is None:
                    continue
                pred = r.predicted.primary_intent or "conceptual"
                exp = r.expected.primary_intent
                if pred in label_to_idx and exp in label_to_idx:
                    matrix[label_to_idx[exp], label_to_idx[pred]] += 1

        else:
            raise ValueError(f"Unknown field: {field}")

        return matrix, labels
