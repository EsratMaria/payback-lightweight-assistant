# PAYBACK Lightweight Assistant

## Overview

Backend microservice that accepts natural-language queries in German or English, classifies
intent via an LLM (Claude or Gemini), retrieves relevant products from a local ChromaDB
vector store, re-ranks them with a loyalty-aware scoring function, and returns structured
JSON — powering the in-app product discovery feature for the PAYBACK loyalty programme.

## Implementation status

| Step | What | Status |
|------|------|--------|
| 1 | Schemas & Pydantic v2 models (`schemas.py`) | Done |
| 2 | Synthetic catalog generation (~700 products, 3 partners) | Done |
| 3 | Retrieval layer — Embedder, ChromaDB, ingest | Done |
| 4 | LLM clients — ClaudeClient (tool-use), GeminiClient (stub) | Done |
| 5 | Intent agent — classification + prompt engineering | Done |
| 5 | Router + clarification agent | Pending |
| 6 | Loyalty ranker | Pending |
| 7 | API routes (`POST /assist`) | Pending |
| 8 | Full test coverage | Partial |
| 9 | Docker / Cloud Run | Pending |

## Architecture

See [docs/decisions.md](docs/decisions.md) for Architecture Decision Records.

## Quick Start

```bash
# 1. Install dependencies (Python 3.13+)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure keys
cp .env.example .env
# Edit .env — set UNIFIED_ENDPOINT_KEY and UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC

# 3. Generate product catalogs (~700 products via Claude)
python data/generate_catalogs.py

# 4. Ingest catalogs into ChromaDB
python -m app.retrieval.ingest

# 5. Run tests
pytest tests/ -v

# 6. Run the API
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

## Key components

### Retrieval (`app/retrieval/`)
- **Embedder** — `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, handles EN + DE queries)
- **LocalChromaStore** — ChromaDB with cosine similarity; similarity = `1.0 − distance`, clamped to [0, 1]
- **Ingest** — idempotent upsert from JSON catalogs; embeds `name + description`, stores `category` as metadata for filtering

### LLM clients (`app/llm/`)
- **ClaudeClient** — forced structured output via Anthropic tool-use (`tool_choice={"type":"tool","name":"respond"}`); Pydantic validation + one retry with warning log on failure
- **GeminiClient** — stub with full implementation sketch in module docstring; toggle via `DEFAULT_LLM_PROVIDER=gemini`

### Intent agent (`app/agents/intent_agent.py`)
Classifies each query into a fully typed `IntentResult`:
- `language` — `"de"` or `"en"`
- `intent` — `"search"`, `"discovery"`, `"comparison"`, `"support"`
- `specificity` — `"specific"`, `"vague"`, `"navigational"`
- `target_partner` — set when a recognized partner (dm, edeka, amazon) is named, regardless of specificity; `null` when multiple recognized partners are mentioned or an unsupported partner (e.g. REWE, Lidl) is named
- `confidence`, `extracted_query`, `reasoning`

## Design Decisions

See [docs/decisions.md](docs/decisions.md) for all Architecture Decision Records.

## Cloud Deployment

Run `./scripts/deploy_gcp.sh <PROJECT_ID>` after setting up GCP credentials.

## Cost & Performance

See `scripts/cost_analysis.py` for per-provider cost estimates and `scripts/load_test.py`
for latency benchmarks.
