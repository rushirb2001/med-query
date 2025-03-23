#!/usr/bin/env python3
"""Explore query routing concepts for medical textbook retrieval.

This script explores different query types and how they might be routed
to different retrieval strategies (vector search, graph traversal, etc.).

Using simple keyword matching for now - will swap this out for LLM-based
intent classification once the routing logic is validated.
"""


def classify_query_intent(query: str) -> str:
    """Classify query intent based on keywords and patterns.

    Simple keyword-based classifier to prototype the routing logic.
    Good enough to test the concept before adding LLM calls.

    Query types:
    - conceptual: "what is", "explain", "describe" -> vector search
    - procedural: "how to", "steps for", "procedure" -> vector + graph
    - relationship: "related to", "connection between" -> graph traversal
    - lookup: specific chapter/section references -> direct lookup
    """
    query_lower = query.lower()

    # Lookup patterns (specific references)
    if any(p in query_lower for p in ["chapter", "section", "page"]):
        return "lookup"

    # Relationship patterns
    if any(p in query_lower for p in ["related", "connection", "between", "compare"]):
        return "relationship"

    # Procedural patterns
    if any(p in query_lower for p in ["how to", "steps", "procedure", "technique"]):
        return "procedural"

    # Default to conceptual
    return "conceptual"


def main():
    """Test query classification with sample queries."""
    test_queries = [
        "What is hemorrhagic shock?",
        "How to perform a thoracotomy?",
        "What is the relationship between trauma and coagulopathy?",
        "Chapter 60 section on wound healing",
        "Explain the physiology of sepsis",
        "Steps for laparoscopic cholecystectomy",
        "Compare open vs minimally invasive surgery",
    ]

    print("Query Intent Classification (keyword-based prototype)")
    print("=" * 60)

    for query in test_queries:
        intent = classify_query_intent(query)
        print(f"Query: {query}")
        print(f"  -> Intent: {intent}")
        print()


if __name__ == "__main__":
    main()
