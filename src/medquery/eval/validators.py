"""Output validation with multiple modes.

Supports:
- STRICT: All fields must match schema exactly
- LENIENT: Allow missing optional fields, coerce types
- PARTIAL: Score partial matches for entities/relationships
"""

import time
from enum import Enum
from typing import Any

from pydantic import ValidationError

from .schemas import (
    QueryAnalysis,
    ValidationResult,
    Entity,
    Relationship,
    ExpectedOutput,
)
from .parsers import JSONExtractor


class ValidationMode(Enum):
    """Validation strictness mode."""

    STRICT = "strict"
    LENIENT = "lenient"
    PARTIAL = "partial"


class OutputValidator:
    """Validate model output against expected schema."""

    def __init__(self, mode: ValidationMode = ValidationMode.LENIENT):
        """Initialize validator.

        Args:
            mode: Validation strictness mode
        """
        self.mode = mode

    def validate(self, raw_output: str) -> ValidationResult:
        """Validate raw model output.

        Args:
            raw_output: Raw string output from model

        Returns:
            ValidationResult with parsed output or errors
        """
        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        # Extract JSON
        parse_result = JSONExtractor.extract(raw_output)

        if parse_result.data is None:
            return ValidationResult(
                valid=False,
                parsed=None,
                errors=[parse_result.error or "Failed to extract JSON"],
                warnings=[],
                raw_output=raw_output,
                parse_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Validate with Pydantic
        try:
            if self.mode == ValidationMode.STRICT:
                parsed = self._validate_strict(parse_result.data, errors)
            else:
                parsed = self._validate_lenient(parse_result.data, errors, warnings)

        except ValidationError as e:
            errors.extend([str(err) for err in e.errors()])
            return ValidationResult(
                valid=False,
                parsed=None,
                errors=errors,
                warnings=warnings,
                raw_output=raw_output,
                parse_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        return ValidationResult(
            valid=len(errors) == 0,
            parsed=parsed,
            errors=errors,
            warnings=warnings,
            raw_output=raw_output,
            parse_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    def _validate_strict(
        self, data: dict[str, Any], errors: list[str]
    ) -> QueryAnalysis | None:
        """Strict validation - all fields must match schema exactly."""
        # Check required fields
        required = ["is_medical", "confidence"]
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        if errors:
            return None

        return QueryAnalysis.model_validate(data)

    def _validate_lenient(
        self, data: dict[str, Any], errors: list[str], warnings: list[str]
    ) -> QueryAnalysis | None:
        """Lenient validation - coerce types, fill defaults."""
        # Coerce is_medical
        if "is_medical" not in data:
            if "medical" in data:
                data["is_medical"] = data.pop("medical")
                warnings.append("Coerced 'medical' to 'is_medical'")
            else:
                errors.append("Missing 'is_medical' field")
                return None

        # Ensure boolean
        if not isinstance(data["is_medical"], bool):
            original = data["is_medical"]
            if isinstance(original, str):
                data["is_medical"] = original.lower() in ("true", "yes", "1")
            elif isinstance(original, (int, float)):
                data["is_medical"] = bool(original)
            warnings.append(f"Coerced is_medical from {type(original).__name__}")

        # Coerce confidence
        if "confidence" not in data:
            data["confidence"] = 0.5
            warnings.append("Missing confidence, defaulting to 0.5")
        else:
            try:
                data["confidence"] = float(data["confidence"])
                data["confidence"] = max(0.0, min(1.0, data["confidence"]))
            except (TypeError, ValueError):
                data["confidence"] = 0.5
                warnings.append("Invalid confidence value, defaulting to 0.5")

        # Coerce intent
        if "primary_intent" not in data and "intent" in data:
            data["primary_intent"] = data.pop("intent")
            warnings.append("Coerced 'intent' to 'primary_intent'")

        # Normalize intent values
        intent_map = {
            "concept": "conceptual",
            "procedure": "procedural",
            "relation": "relationship",
            "reference": "lookup",
        }
        if data.get("primary_intent") in intent_map:
            data["primary_intent"] = intent_map[data["primary_intent"]]
            warnings.append("Normalized primary_intent value")

        # Validate entities format
        if "entities" in data:
            data["entities"] = self._coerce_entities(data["entities"], warnings)

        # Validate relationships format
        if "relationships" in data:
            data["relationships"] = self._coerce_relationships(
                data["relationships"], warnings
            )

        return QueryAnalysis.model_validate(data)

    def _coerce_entities(
        self, entities: Any, warnings: list[str]
    ) -> list[dict[str, Any]]:
        """Coerce entities to valid format."""
        if not isinstance(entities, list):
            warnings.append("Entities is not a list, returning empty")
            return []

        valid_entities = []
        valid_types = {"condition", "procedure", "anatomy", "process", "concept", "medication"}

        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                warnings.append(f"Entity {i} is not a dict, skipping")
                continue

            if "text" not in entity:
                warnings.append(f"Entity {i} missing 'text', skipping")
                continue

            # Coerce type
            entity_type = entity.get("type", "concept")
            if entity_type not in valid_types:
                entity["type"] = "concept"
                warnings.append(f"Entity {i} invalid type '{entity_type}', defaulting to 'concept'")

            valid_entities.append(entity)

        return valid_entities

    def _coerce_relationships(
        self, relationships: Any, warnings: list[str]
    ) -> list[dict[str, Any]]:
        """Coerce relationships to valid format."""
        if not isinstance(relationships, list):
            warnings.append("Relationships is not a list, returning empty")
            return []

        valid_rels = []
        valid_types = {"affects", "causes", "treats", "indicates", "has_property", "compared_to", "part_of", "precedes"}

        for i, rel in enumerate(relationships):
            if not isinstance(rel, dict):
                warnings.append(f"Relationship {i} is not a dict, skipping")
                continue

            if "source" not in rel or "target" not in rel:
                warnings.append(f"Relationship {i} missing source/target, skipping")
                continue

            # Coerce type
            rel_type = rel.get("type", "affects")
            if rel_type not in valid_types:
                rel["type"] = "affects"
                warnings.append(f"Relationship {i} invalid type '{rel_type}', defaulting to 'affects'")

            valid_rels.append(rel)

        return valid_rels

    async def validate_batch(
        self, outputs: list[str]
    ) -> list[ValidationResult]:
        """Validate multiple outputs.

        Args:
            outputs: List of raw outputs

        Returns:
            List of validation results
        """
        return [self.validate(output) for output in outputs]


class AccuracyScorer:
    """Score predicted output against expected output."""

    @staticmethod
    def score_medical(predicted: QueryAnalysis, expected: ExpectedOutput) -> bool:
        """Check if medical classification matches."""
        return predicted.is_medical == expected.is_medical

    @staticmethod
    def score_intent(predicted: QueryAnalysis, expected: ExpectedOutput) -> bool:
        """Check if primary intent matches."""
        if expected.primary_intent is None:
            return True  # No expected intent to check
        return predicted.primary_intent == expected.primary_intent

    @staticmethod
    def score_entities(
        predicted: QueryAnalysis, expected: ExpectedOutput
    ) -> tuple[float, float, float]:
        """Calculate entity precision, recall, F1.

        Returns:
            Tuple of (precision, recall, f1)
        """
        if not expected.entities:
            return (1.0, 1.0, 1.0) if not predicted.entities else (0.0, 1.0, 0.0)

        if not predicted.entities:
            return (0.0, 0.0, 0.0)

        # Convert to sets for comparison (case-insensitive text match)
        pred_set = {(e.text.lower(), e.type) for e in predicted.entities}
        exp_set = {(e.text.lower(), e.type) for e in expected.entities}

        true_positives = len(pred_set & exp_set)
        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(exp_set) if exp_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return (precision, recall, f1)

    @staticmethod
    def score_relationships(
        predicted: QueryAnalysis, expected: ExpectedOutput
    ) -> tuple[float, float, float]:
        """Calculate relationship precision, recall, F1.

        Returns:
            Tuple of (precision, recall, f1)
        """
        if not expected.relationships:
            return (1.0, 1.0, 1.0) if not predicted.relationships else (0.0, 1.0, 0.0)

        if not predicted.relationships:
            return (0.0, 0.0, 0.0)

        # Convert to sets for comparison
        pred_set = {
            (r.source.lower(), r.target.lower(), r.type)
            for r in predicted.relationships
        }
        exp_set = {
            (r.source.lower(), r.target.lower(), r.type)
            for r in expected.relationships
        }

        true_positives = len(pred_set & exp_set)
        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(exp_set) if exp_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return (precision, recall, f1)

    @staticmethod
    def score_all(
        predicted: QueryAnalysis, expected: ExpectedOutput
    ) -> dict[str, Any]:
        """Calculate all scores.

        Returns:
            Dictionary with all scores
        """
        entity_scores = AccuracyScorer.score_entities(predicted, expected)
        rel_scores = AccuracyScorer.score_relationships(predicted, expected)

        return {
            "medical_correct": AccuracyScorer.score_medical(predicted, expected),
            "intent_correct": AccuracyScorer.score_intent(predicted, expected),
            "entity_precision": entity_scores[0],
            "entity_recall": entity_scores[1],
            "entity_f1": entity_scores[2],
            "relationship_precision": rel_scores[0],
            "relationship_recall": rel_scores[1],
            "relationship_f1": rel_scores[2],
            "confidence": predicted.confidence,
            "expected_confidence": expected.confidence,
        }
