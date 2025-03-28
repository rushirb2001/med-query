"""Validation and deduplication utilities for generated queries.

Provides quality checks, deduplication, and statistics for the generated dataset.
"""

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional


def validate_query(query_dict: dict) -> tuple[bool, list[str]]:
    """Validate a single query for required fields and consistency.

    Args:
        query_dict: Query dictionary to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Required top-level fields
    required = ["id", "query", "level", "expected", "category"]
    for field in required:
        if field not in query_dict:
            errors.append(f"Missing required field: {field}")

    if "query" in query_dict:
        query = query_dict["query"]
        if not query or len(query) < 5:
            errors.append("Query too short (< 5 chars)")
        if len(query) > 500:
            errors.append("Query too long (> 500 chars)")

    # Validate expected fields
    if "expected" in query_dict:
        expected = query_dict["expected"]

        if "is_medical" not in expected:
            errors.append("Missing is_medical in expected")
        elif not isinstance(expected["is_medical"], bool):
            errors.append("is_medical must be boolean")

        if "confidence" in expected:
            conf = expected["confidence"]
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                errors.append("confidence must be float between 0 and 1")

        if "primary_intent" in expected and expected["is_medical"]:
            valid_intents = ["conceptual", "procedural", "relationship", "lookup"]
            if expected["primary_intent"] not in valid_intents:
                errors.append(f"Invalid primary_intent: {expected['primary_intent']}")

        # Validate entities
        if "entities" in expected:
            valid_entity_types = ["condition", "procedure", "anatomy", "process", "concept", "medication"]
            for i, entity in enumerate(expected["entities"]):
                if "text" not in entity:
                    errors.append(f"Entity {i} missing text")
                if "type" not in entity:
                    errors.append(f"Entity {i} missing type")
                elif entity["type"] not in valid_entity_types:
                    errors.append(f"Entity {i} has invalid type: {entity['type']}")

        # Validate relationships
        if "relationships" in expected:
            valid_rel_types = ["affects", "causes", "treats", "indicates", "has_property", "compared_to", "part_of", "precedes"]
            for i, rel in enumerate(expected["relationships"]):
                if "source" not in rel:
                    errors.append(f"Relationship {i} missing source")
                if "target" not in rel:
                    errors.append(f"Relationship {i} missing target")
                if "type" not in rel:
                    errors.append(f"Relationship {i} missing type")
                elif rel["type"] not in valid_rel_types:
                    errors.append(f"Relationship {i} has invalid type: {rel['type']}")

    # Validate level
    if "level" in query_dict:
        valid_levels = ["L1", "L2", "L3", "L4", "L5"]
        if query_dict["level"] not in valid_levels:
            errors.append(f"Invalid level: {query_dict['level']}")

    return len(errors) == 0, errors


def similarity_ratio(s1: str, s2: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def deduplicate_queries(
    queries: list[dict],
    similarity_threshold: float = 0.85,
) -> tuple[list[dict], list[dict]]:
    """Remove duplicate and near-duplicate queries.

    Args:
        queries: List of query dictionaries
        similarity_threshold: Threshold for considering queries as duplicates

    Returns:
        Tuple of (unique queries, removed duplicates)
    """
    unique = []
    duplicates = []
    seen_queries = []

    for query_dict in queries:
        query_text = query_dict.get("query", "").strip().lower()

        # Check exact duplicates
        if query_text in seen_queries:
            duplicates.append(query_dict)
            continue

        # Check near-duplicates
        is_duplicate = False
        for seen in seen_queries:
            if similarity_ratio(query_text, seen) > similarity_threshold:
                is_duplicate = True
                duplicates.append(query_dict)
                break

        if not is_duplicate:
            unique.append(query_dict)
            seen_queries.append(query_text)

    return unique, duplicates


def validate_dataset(filepath: str) -> dict:
    """Validate an entire dataset file.

    Args:
        filepath: Path to JSON dataset file

    Returns:
        Validation report dictionary
    """
    with open(filepath) as f:
        queries = json.load(f)

    report = {
        "total": len(queries),
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "by_level": Counter(),
        "by_category": Counter(),
        "by_intent": Counter(),
        "medical_count": 0,
        "non_medical_count": 0,
        "has_entities": 0,
        "has_relationships": 0,
        "has_decomposition": 0,
    }

    for q in queries:
        is_valid, errors = validate_query(q)

        if is_valid:
            report["valid"] += 1
        else:
            report["invalid"] += 1
            report["errors"].append({"id": q.get("id", "unknown"), "errors": errors})

        # Collect stats
        report["by_level"][q.get("level", "unknown")] += 1
        report["by_category"][q.get("category", "unknown")] += 1

        if "expected" in q:
            exp = q["expected"]
            if exp.get("is_medical"):
                report["medical_count"] += 1
                report["by_intent"][exp.get("primary_intent", "unknown")] += 1
            else:
                report["non_medical_count"] += 1

            if exp.get("entities"):
                report["has_entities"] += 1
            if exp.get("relationships"):
                report["has_relationships"] += 1
            if exp.get("decomposition"):
                report["has_decomposition"] += 1

    return report


def print_validation_report(report: dict):
    """Print a formatted validation report."""
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print(f"\nTotal queries: {report['total']}")
    print(f"Valid: {report['valid']} ({report['valid']/report['total']*100:.1f}%)")
    print(f"Invalid: {report['invalid']}")

    print(f"\nMedical: {report['medical_count']}")
    print(f"Non-medical: {report['non_medical_count']}")

    print("\nBy Level:")
    for level, count in sorted(report["by_level"].items()):
        print(f"  {level}: {count}")

    print("\nBy Intent:")
    for intent, count in sorted(report["by_intent"].items()):
        print(f"  {intent}: {count}")

    print("\nBy Category:")
    for cat, count in sorted(report["by_category"].items()):
        print(f"  {cat}: {count}")

    print(f"\nWith entities: {report['has_entities']}")
    print(f"With relationships: {report['has_relationships']}")
    print(f"With decomposition: {report['has_decomposition']}")

    if report["errors"]:
        print(f"\nFirst 5 errors:")
        for err in report["errors"][:5]:
            print(f"  {err['id']}: {', '.join(err['errors'])}")


def merge_datasets(filepaths: list[str], output_path: str):
    """Merge multiple dataset files into one.

    Args:
        filepaths: List of paths to merge
        output_path: Output file path
    """
    all_queries = []
    for fp in filepaths:
        with open(fp) as f:
            queries = json.load(f)
            all_queries.extend(queries)
            print(f"Loaded {len(queries)} from {fp}")

    # Deduplicate
    unique, duplicates = deduplicate_queries(all_queries)
    print(f"Removed {len(duplicates)} duplicates")

    # Reassign IDs
    for i, q in enumerate(unique):
        q["id"] = f"MERGED-{i+1:05d}"

    with open(output_path, "w") as f:
        json.dump(unique, f, indent=2)

    print(f"Saved {len(unique)} queries to {output_path}")


def get_dataset_stats(data_dir: str) -> dict:
    """Get statistics across all dataset files in a directory.

    Args:
        data_dir: Path to data directory

    Returns:
        Combined statistics dictionary
    """
    data_path = Path(data_dir)
    stats = {
        "total": 0,
        "by_file": {},
        "by_level": Counter(),
        "by_intent": Counter(),
        "medical": 0,
        "non_medical": 0,
    }

    for json_file in data_path.glob("*.json"):
        with open(json_file) as f:
            queries = json.load(f)

        stats["by_file"][json_file.name] = len(queries)
        stats["total"] += len(queries)

        for q in queries:
            stats["by_level"][q.get("level", "unknown")] += 1
            if "expected" in q:
                if q["expected"].get("is_medical"):
                    stats["medical"] += 1
                    stats["by_intent"][q["expected"].get("primary_intent", "unknown")] += 1
                else:
                    stats["non_medical"] += 1

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        report = validate_dataset(filepath)
        print_validation_report(report)
    else:
        print("Usage: python validate.py <dataset.json>")
