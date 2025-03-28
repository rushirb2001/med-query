#!/usr/bin/env python3
"""Generate P2 (Priority 2) dataset - 1500 queries.

P2 Categories:
- medical_lookup: 500 queries
- entity_extraction: 500 queries
- relationship_extraction: 500 queries

Usage:
    python scripts/gen_p2_dataset.py [--category CATEGORY] [--count COUNT] [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generators.base import QueryGenerator, GeneratedQuery
from generators.templates import QUERY_TEMPLATES


# P2 Dataset configuration
P2_CONFIG = {
    "medical_lookup": {
        "count": 500,
        "template": QUERY_TEMPLATES["medical_lookup"],
        "level": "L1",
        "is_medical": True,
        "intent": "lookup",
        "output": "data/medical_lookup.json",
    },
    "entity_extraction": {
        "count": 500,
        "template": QUERY_TEMPLATES["entity_extraction"],
        "level": "L3",
        "is_medical": True,
        "intent": None,  # Mixed intents, focus on entity richness
        "output": "data/entity_extraction.json",
    },
    "relationship_extraction": {
        "count": 500,
        "template": QUERY_TEMPLATES["relationship_extraction"],
        "level": "L3",
        "is_medical": True,
        "intent": "relationship",
        "output": "data/relationship_extraction.json",
    },
}


def generate_category(
    category: str,
    config: dict,
    generator: QueryGenerator,
    scripts_dir: Path,
    dry_run: bool = False,
    count_override: int = None,
) -> list[GeneratedQuery]:
    """Generate queries for a single category."""
    count = count_override or config["count"]
    if dry_run:
        count = min(5, count)

    print(f"\n{'=' * 60}")
    print(f"Generating: {category}")
    print(f"Count: {count}")
    print(f"Level: {config['level']}")
    print("=" * 60)

    queries = generator.generate_queries(
        category=category,
        count=count,
        template=config["template"],
        level=config["level"],
        is_medical=config["is_medical"],
        intent=config["intent"],
        batch_size=25,
    )

    output_path = scripts_dir / config["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator.save_queries(queries, str(output_path))

    print(f"Generated {len(queries)} queries")

    # Show sample
    if queries:
        print("\nSample queries:")
        for q in queries[:3]:
            print(f"  - {q.query[:60]}...")
            if q.entities:
                print(f"    Entities: {[e.text for e in q.entities[:3]]}")
            if q.relationships:
                print(f"    Relations: {[(r.source, r.type, r.target) for r in q.relationships[:2]]}")

    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate P2 dataset")
    parser.add_argument(
        "--category",
        choices=list(P2_CONFIG.keys()) + ["all"],
        default="all",
        help="Category to generate (default: all)",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Override query count (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate only 5 queries per category for testing",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        help="Claude model to use (default: from ANTHROPIC_MODEL env var)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable prompt caching",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    scripts_dir = Path(__file__).parent
    generator = QueryGenerator(model=args.model, use_cache=not args.no_cache)

    categories = [args.category] if args.category != "all" else list(P2_CONFIG.keys())

    all_queries = []
    for category in categories:
        config = P2_CONFIG[category]
        queries = generate_category(
            category=category,
            config=config,
            generator=generator,
            scripts_dir=scripts_dir,
            dry_run=args.dry_run,
            count_override=args.count,
        )
        all_queries.extend(queries)

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total queries: {len(all_queries)}")

    # Entity/relationship stats
    with_entities = sum(1 for q in all_queries if q.entities)
    with_relationships = sum(1 for q in all_queries if q.relationships)
    print(f"\nWith entities: {with_entities}")
    print(f"With relationships: {with_relationships}")


if __name__ == "__main__":
    main()
