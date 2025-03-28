#!/usr/bin/env python3
"""Generate P1 (Priority 1) dataset - 8500 queries.

P1 Categories:
- medical_conceptual: 1500 queries
- medical_procedural: 1500 queries
- medical_relationship: 1000 queries
- medical_complex: 1500 queries
- non_medical: 2000 queries
- edge_cases: 1000 queries

Usage:
    python scripts/gen_p1_dataset.py [--category CATEGORY] [--count COUNT] [--dry-run]
    python scripts/gen_p1_dataset.py --async --concurrency 10  # Async mode
"""

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generators.base import QueryGenerator, GeneratedQuery
from generators.templates import QUERY_TEMPLATES


# P1 Dataset configuration
P1_CONFIG = {
    # "medical_conceptual": {
    #     "count": 1500,
    #     "template": QUERY_TEMPLATES["medical_conceptual"],
    #     "level": "L1",
    #     "is_medical": True,
    #     "intent": "conceptual",
    #     "output": "data/medical_conceptual.json",
    # },
    # "medical_procedural": {
    #     "count": 1500,
    #     "template": QUERY_TEMPLATES["medical_procedural"],
    #     "level": "L1",
    #     "is_medical": True,
    #     "intent": "procedural",
    #     "output": "data/medical_procedural.json",
    # },
    "medical_relationship": {
        "count": 1000,
        "template": QUERY_TEMPLATES["medical_relationship"],
        "level": "L2",
        "is_medical": True,
        "intent": "relationship",
        "output": "data/medical_relationship.json",
    },
    "medical_complex": {
        "count": 1500,
        "template": QUERY_TEMPLATES["medical_complex"],
        "level": "L4",
        "is_medical": True,
        "intent": None,  # Mixed intents
        "output": "data/medical_complex.json",
    },
    "non_medical": {
        "count": 2000,
        "template": QUERY_TEMPLATES["non_medical"],
        "level": "L1",
        "is_medical": False,
        "intent": None,
        "output": "data/non_medical.json",
    },
    "edge_cases": {
        "count": 1000,
        "template": QUERY_TEMPLATES["edge_cases"],
        "level": "L2",
        "is_medical": None,  # Mixed
        "intent": None,
        "output": "data/edge_cases.json",
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
    """Generate queries for a single category (sync).

    Args:
        category: Category name
        config: Category configuration
        generator: QueryGenerator instance
        scripts_dir: Path to scripts directory
        dry_run: If True, generate only 5 queries for testing
        count_override: Override the count from config

    Returns:
        List of generated queries
    """
    count = count_override or config["count"]
    if dry_run:
        count = min(5, count)

    print(f"\n{'=' * 60}")
    print(f"Generating: {category}")
    print(f"Count: {count}")
    print(f"Level: {config['level']}")
    print(f"Medical: {config['is_medical']}")
    print(f"Intent: {config['intent']}")
    print("=" * 60)

    # Use smaller batch size for complex categories (more tokens per query)
    # Haiku has 4096 max output, ~300 tokens per query = ~12 queries max
    batch_size = config.get("batch_size", 10)

    queries = generator.generate_queries(
        category=category,
        count=count,
        template=config["template"],
        level=config["level"],
        is_medical=config["is_medical"] if config["is_medical"] is not None else True,
        intent=config["intent"],
        batch_size=batch_size,
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

    return queries


async def generate_category_async(
    category: str,
    config: dict,
    generator: QueryGenerator,
    scripts_dir: Path,
    dry_run: bool = False,
    count_override: int = None,
) -> list[GeneratedQuery]:
    """Generate queries for a single category (async).

    Args:
        category: Category name
        config: Category configuration
        generator: QueryGenerator instance
        scripts_dir: Path to scripts directory
        dry_run: If True, generate only 5 queries for testing
        count_override: Override the count from config

    Returns:
        List of generated queries
    """
    count = count_override or config["count"]
    if dry_run:
        count = min(5, count)

    print(f"\n{'=' * 60}")
    print(f"Generating: {category} (async)")
    print(f"Count: {count}")
    print(f"Level: {config['level']}")
    print(f"Medical: {config['is_medical']}")
    print(f"Intent: {config['intent']}")
    print("=" * 60)

    # Use smaller batch size for complex categories (more tokens per query)
    # Haiku has 4096 max output, ~300 tokens per query = ~12 queries max
    batch_size = config.get("batch_size", 10)

    queries = await generator.generate_queries_async(
        category=category,
        count=count,
        template=config["template"],
        level=config["level"],
        is_medical=config["is_medical"] if config["is_medical"] is not None else True,
        intent=config["intent"],
        batch_size=batch_size,
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

    return queries


async def main_async(args, scripts_dir: Path, categories: list[str]):
    """Async main function for parallel generation."""
    generator = QueryGenerator(
        model=args.model,
        use_cache=not args.no_cache,
        max_concurrency=args.concurrency,
    )

    all_queries = []
    for category in categories:
        config = P1_CONFIG[category]
        queries = await generate_category_async(
            category=category,
            config=config,
            generator=generator,
            scripts_dir=scripts_dir,
            dry_run=args.dry_run,
            count_override=args.count,
        )
        all_queries.extend(queries)

    return all_queries


def main():
    parser = argparse.ArgumentParser(description="Generate P1 dataset")
    parser.add_argument(
        "--category",
        choices=list(P1_CONFIG.keys()) + ["all"],
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
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async mode for parallel batch generation",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API calls in async mode (default: 10)",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-key-here")
        sys.exit(1)

    scripts_dir = Path(__file__).parent
    categories = [args.category] if args.category != "all" else list(P1_CONFIG.keys())

    if args.use_async:
        print(f"Running in ASYNC mode with concurrency={args.concurrency}")
        all_queries = asyncio.run(main_async(args, scripts_dir, categories))
    else:
        generator = QueryGenerator(model=args.model, use_cache=not args.no_cache)
        all_queries = []
        for category in categories:
            config = P1_CONFIG[category]
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

    # Stats by category
    by_category = Counter(q.category for q in all_queries)
    print("\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")

    by_level = Counter(q.level for q in all_queries)
    print("\nBy level:")
    for level, count in sorted(by_level.items()):
        print(f"  {level}: {count}")

    medical_count = sum(1 for q in all_queries if q.is_medical)
    print(f"\nMedical: {medical_count}")
    print(f"Non-medical: {len(all_queries) - medical_count}")


if __name__ == "__main__":
    main()
