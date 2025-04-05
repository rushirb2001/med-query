# MedQuery

**LangGraph-based query routing and orchestration for medical textbook retrieval.**

## Overview

MedQuery is an intelligent query router that classifies user intent and orchestrates retrieval from multiple sources. Built with LangGraph for workflow management and designed to integrate with hybrid vector-graph retrieval systems.

## Features

- Intent classification (conceptual, procedural, relationship, lookup)
- Multi-agent workflow (router → retriever → synthesizer)
- Conversation state management
- Tool calling with structured outputs
- Citation formatting
- Medical domain boundary detection with disambiguation
- Entity extraction (condition, procedure, anatomy, process, concept, medication)
- Relationship extraction (affects, causes, treats, indicates, compared_to)
- Query decomposition for multi-intent queries
- Benchmark evaluation framework with configurable validation modes
- LLM-assisted dataset generation with Anthropic prompt caching

## Installation

```bash
poetry install
```

## Quick Start

```python
from med_query import MedQuery

mq = MedQuery()
response = mq.query("What is hemorrhagic shock?")
print(response)
```

## Architecture

```
User Query → Router Agent → Retriever Agent → Synthesizer Agent → Response
                 │                │
                 ▼                ▼
           Intent Class     Tool Selection
```

## Module Structure

| Module | Description |
|--------|-------------|
| `medquery.types` | Shared type definitions (EntityType, RelationshipType, IntentType) |
| `medquery.backends` | Inference backends (MLX, OpenAI, Anthropic) with prompt presets |
| `medquery.eval` | Benchmark evaluation framework with metrics collection |
| `medquery.generators` | LLM-assisted dataset generation with validation utilities |

## Evaluation Framework

The evaluation subsystem provides comprehensive benchmarking capabilities:

- **Validation Modes**: strict, lenient, partial output validation
- **Metrics**: medical accuracy, intent accuracy, entity F1, latency percentiles
- **Reports**: JSON, Markdown, HTML, CSV export formats
- **Backends**: MLX (Apple Silicon), OpenAI API, Anthropic API

## License

**PROPRIETARY AND CONFIDENTIAL**

Copyright (c) 2025 Rushir Bhavsar. All Rights Reserved.
