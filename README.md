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

## License

**PROPRIETARY AND CONFIDENTIAL**

Copyright (c) 2025 Rushir Bhavsar. All Rights Reserved.
