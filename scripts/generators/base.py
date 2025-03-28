"""Base query generator with Claude API integration.

Provides LLM-assisted generation of diverse medical and non-medical queries
with full annotations for classification, entity extraction, and relationships.

Uses Anthropic prompt caching to reduce costs when generating large datasets.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from anthropic import Anthropic, AsyncAnthropic
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# Available models (as of Jan 2026):
# - claude-sonnet-4-20250514 (best quality/cost balance)
# - claude-opus-4-20250514 (highest quality)
# - claude-3-5-haiku-20241022 (fast, cheaper)
# - claude-3-haiku-20240307 (fastest, cheapest)
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


@dataclass
class Entity:
    """Medical entity with optional UMLS CUI."""
    text: str
    type: str  # condition, procedure, anatomy, process, concept, medication
    cui: Optional[str] = None  # UMLS Concept Unique Identifier


@dataclass
class Relationship:
    """Relationship between two entities."""
    source: str
    target: str
    type: str  # affects, causes, treats, indicates, has_property, compared_to


@dataclass
class SubQuery:
    """Decomposed sub-query."""
    query: str
    intent: str


@dataclass
class GeneratedQuery:
    """Full query with all annotations."""
    id: str
    query: str
    level: str  # L1-L5 complexity
    is_medical: bool
    confidence: float
    primary_intent: Optional[str]  # conceptual, procedural, relationship, lookup
    secondary_intent: Optional[str] = None
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    decomposition: list[SubQuery] = field(default_factory=list)
    retrieval_strategy: list[str] = field(default_factory=list)
    category: str = ""  # For grouping (trauma, surgery, non_medical_tech, etc.)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "query": self.query,
            "level": self.level,
            "expected": {
                "is_medical": self.is_medical,
                "confidence": self.confidence,
                "primary_intent": self.primary_intent,
                "secondary_intent": self.secondary_intent,
                "entities": [asdict(e) for e in self.entities],
                "relationships": [asdict(r) for r in self.relationships],
                "decomposition": [asdict(s) for s in self.decomposition],
                "retrieval_strategy": self.retrieval_strategy,
            },
            "category": self.category,
        }


class QueryGenerator:
    """LLM-assisted query generator using Claude API with prompt caching."""

    def __init__(self, model: str = DEFAULT_MODEL, use_cache: bool = True, max_concurrency: int = 10):
        """Initialize generator with Claude client.

        Args:
            model: Claude model to use for generation
            use_cache: Whether to use prompt caching (reduces cost for repeated prompts)
            max_concurrency: Max concurrent API calls for async generation
        """
        self.client = Anthropic()
        self.async_client = AsyncAnthropic()
        self.model = model
        self.use_cache = use_cache
        self.max_concurrency = max_concurrency
        self.generated_count = 0
        self._cached_templates: dict[str, str] = {}  # Cache template hashes

    def generate_queries(
        self,
        category: str,
        count: int,
        template: str,
        level: str = "L1",
        is_medical: bool = True,
        intent: Optional[str] = None,
        batch_size: int = 20,
    ) -> list[GeneratedQuery]:
        """Generate queries using Claude API.

        Args:
            category: Category name for grouping
            count: Number of queries to generate
            template: Prompt template for generation
            level: Complexity level (L1-L5)
            is_medical: Whether queries should be medical
            intent: Primary intent type
            batch_size: Queries per API call

        Returns:
            List of generated queries
        """
        queries = []
        batches = (count + batch_size - 1) // batch_size

        with tqdm(total=count, desc=f"Generating {category}", unit="queries") as pbar:
            for batch in range(batches):
                remaining = min(batch_size, count - len(queries))
                if remaining <= 0:
                    break

                batch_queries = self._generate_batch(
                    category=category,
                    count=remaining,
                    template=template,
                    level=level,
                    is_medical=is_medical,
                    intent=intent,
                    start_id=self.generated_count,
                )
                queries.extend(batch_queries)
                self.generated_count += len(batch_queries)
                pbar.update(len(batch_queries))

                # Rate limiting
                if batch < batches - 1:
                    time.sleep(0.5)

        return queries

    def _generate_batch(
        self,
        category: str,
        count: int,
        template: str,
        level: str,
        is_medical: bool,
        intent: Optional[str],
        start_id: int,
        retry_count: int = 0,
    ) -> list[GeneratedQuery]:
        """Generate a batch of queries with prompt caching."""
        max_retries = 2

        tqdm.write(f"  → Building prompts...")
        system_prompt = self._build_system_prompt(template)
        user_prompt = self._build_user_prompt(
            category=category,
            count=count,
            level=level,
            is_medical=is_medical,
            intent=intent,
        )
        tqdm.write(f"  ✓ Prompts ready (system: {len(system_prompt)} chars, user: {len(user_prompt)} chars)")

        # Use prompt caching for the system prompt (template stays the same)
        # Haiku models have 4096 max, Sonnet/Opus have 8192
        max_tokens = 4096 if "haiku" in self.model else 8192

        tqdm.write(f"  → Calling API for batch of {count}...")
        if self.use_cache:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

        # Log cache and token usage
        if hasattr(response, "usage"):
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            cached = getattr(usage, "cache_read_input_tokens", 0)
            cache_str = f", cache: {cached}" if cached > 0 else ""
            tqdm.write(f"  ✓ Received response (in: {input_tokens}, out: {output_tokens}{cache_str})")

            # Warn if output might be truncated
            if output_tokens >= max_tokens - 100:
                tqdm.write(f"  ⚠ Response may be truncated ({output_tokens}/{max_tokens} tokens)")

        queries = self._parse_response(
            response.content[0].text,
            category=category,
            level=level,
            is_medical=is_medical,
            intent=intent,
            start_id=start_id,
        )

        # Retry with smaller batch if parse failed
        if len(queries) == 0 and retry_count < max_retries:
            smaller_count = max(5, count // 2)
            tqdm.write(f"  ⚠ Parse failed, retrying with smaller batch ({smaller_count})...")
            time.sleep(1)
            return self._generate_batch(
                category=category,
                count=smaller_count,
                template=template,
                level=level,
                is_medical=is_medical,
                intent=intent,
                start_id=start_id,
                retry_count=retry_count + 1,
            )

        return queries

    def _build_system_prompt(self, template: str) -> str:
        """Build system prompt (cacheable)."""
        return f"""{template}

You are a query generator for a medical textbook retrieval system test dataset.
Generate diverse, realistic queries with full annotations.

Output format - JSON array with objects containing:
- query: The query text
- confidence: 0.0-1.0 confidence score
- primary_intent: conceptual|procedural|relationship|lookup|null
- secondary_intent: optional secondary intent
- entities: array of {{text, type, cui}} where type is condition|procedure|anatomy|process|concept|medication
- relationships: array of {{source, target, type}} where type is affects|causes|treats|indicates|compared_to
- decomposition: array of {{query, intent}} for complex queries needing breakdown
- retrieval_strategy: array of vector_search|graph_traversal|hybrid_search|metadata_lookup

Return ONLY valid JSON array, no markdown or explanations."""

    def _build_user_prompt(
        self,
        category: str,
        count: int,
        level: str,
        is_medical: bool,
        intent: Optional[str],
    ) -> str:
        """Build user prompt (varies per batch)."""
        intent_str = intent or "varied"
        medical_str = "medical" if is_medical else "non-medical"

        return f"""Generate exactly {count} diverse {medical_str} queries for category "{category}".

Requirements:
- Complexity Level: {level}
- Medical: {is_medical}
- Primary Intent: {intent_str}
- Each query must be unique and realistic
- Include entity annotations where applicable
- For complex queries (L3+), include relationship annotations

Return JSON array now:"""

    def _parse_response(
        self,
        response: str,
        category: str,
        level: str,
        is_medical: bool,
        intent: Optional[str],
        start_id: int,
    ) -> list[GeneratedQuery]:
        """Parse Claude response into GeneratedQuery objects."""
        queries = []

        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            data = json.loads(response)

            for i, item in enumerate(data):
                query_id = f"{category.upper()}-{start_id + i + 1:05d}"

                entities = [
                    Entity(
                        text=e.get("text", ""),
                        type=e.get("type", "concept"),
                        cui=e.get("cui"),
                    )
                    for e in item.get("entities", [])
                ]

                relationships = [
                    Relationship(
                        source=r.get("source", ""),
                        target=r.get("target", ""),
                        type=r.get("type", "affects"),
                    )
                    for r in item.get("relationships", [])
                ]

                decomposition = [
                    SubQuery(
                        query=s.get("query", ""),
                        intent=s.get("intent", "conceptual"),
                    )
                    for s in item.get("decomposition", [])
                ]

                queries.append(GeneratedQuery(
                    id=query_id,
                    query=item.get("query", ""),
                    level=level,
                    is_medical=is_medical,
                    confidence=item.get("confidence", 0.9),
                    primary_intent=item.get("primary_intent", intent),
                    secondary_intent=item.get("secondary_intent"),
                    entities=entities,
                    relationships=relationships,
                    decomposition=decomposition,
                    retrieval_strategy=item.get("retrieval_strategy", ["vector_search"]),
                    category=category,
                ))

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response: {response[:500]}")

        return queries

    def save_queries(self, queries: list[GeneratedQuery], filepath: str):
        """Save queries to JSON file."""
        data = [q.to_dict() for q in queries]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(queries)} queries to {filepath}")

    async def generate_queries_async(
        self,
        category: str,
        count: int,
        template: str,
        level: str = "L1",
        is_medical: bool = True,
        intent: Optional[str] = None,
        batch_size: int = 25,
    ) -> list[GeneratedQuery]:
        """Generate queries using Claude API with async concurrency.

        Args:
            category: Category name for grouping
            count: Number of queries to generate
            template: Prompt template for generation
            level: Complexity level (L1-L5)
            is_medical: Whether queries should be medical
            intent: Primary intent type
            batch_size: Queries per API call

        Returns:
            List of generated queries
        """
        batches = (count + batch_size - 1) // batch_size
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Prepare batch configs
        batch_configs = []
        current_id = self.generated_count
        for batch in range(batches):
            batch_count = min(batch_size, count - batch * batch_size)
            if batch_count <= 0:
                break
            batch_configs.append({
                "category": category,
                "count": batch_count,
                "template": template,
                "level": level,
                "is_medical": is_medical,
                "intent": intent,
                "start_id": current_id,
                "batch_num": batch + 1,
                "total_batches": batches,
            })
            current_id += batch_count

        async def run_batch(config: dict) -> list[GeneratedQuery]:
            async with semaphore:
                return await self._generate_batch_async(**config)

        # Run all batches concurrently with progress bar
        results = []
        with tqdm(total=count, desc=f"Generating {category}", unit="queries") as pbar:
            tasks = [run_batch(config) for config in batch_configs]
            for coro in asyncio.as_completed(tasks):
                batch_queries = await coro
                results.extend(batch_queries)
                pbar.update(len(batch_queries))

        self.generated_count = current_id
        return results

    async def _generate_batch_async(
        self,
        category: str,
        count: int,
        template: str,
        level: str,
        is_medical: bool,
        intent: Optional[str],
        start_id: int,
        batch_num: int = 0,
        total_batches: int = 0,
    ) -> list[GeneratedQuery]:
        """Generate a batch of queries asynchronously with prompt caching."""
        system_prompt = self._build_system_prompt(template)
        user_prompt = self._build_user_prompt(
            category=category,
            count=count,
            level=level,
            is_medical=is_medical,
            intent=intent,
        )
        tqdm.write(f"  → [{batch_num}/{total_batches}] Calling API for batch of {count}...")

        # Use prompt caching for the system prompt (template stays the same)
        # Haiku models have 4096 max, Sonnet/Opus have 8192
        max_tokens = 4096 if "haiku" in self.model else 8192

        if self.use_cache:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
        else:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

        # Log cache and token usage
        if hasattr(response, "usage"):
            usage = response.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            cached = getattr(usage, "cache_read_input_tokens", 0)
            cache_str = f", cache: {cached}" if cached > 0 else ""
            tqdm.write(f"  ✓ [{batch_num}/{total_batches}] Done (in: {input_tokens}, out: {output_tokens}{cache_str})")

        return self._parse_response(
            response.content[0].text,
            category=category,
            level=level,
            is_medical=is_medical,
            intent=intent,
            start_id=start_id,
        )


def load_queries(filepath: str) -> list[dict]:
    """Load queries from JSON file."""
    with open(filepath) as f:
        return json.load(f)
