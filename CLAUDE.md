# CLAUDE.md — PAYBACK Lightweight Assistant

This file gives Claude Code the context it needs to work effectively in this repo.

## Project purpose

Backend microservice for the PAYBACK loyalty app's lightweight assistant feature.
Takes a user query (EN/DE), classifies intent via LLM, routes to the right action
(search, clarify, navigate), returns structured JSON with product recommendations.

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI + Pydantic v2 |
| LLM | Anthropic Claude (primary), Google Gemini (secondary) |
| Embeddings | sentence-transformers (local, multilingual) |
| Vector store | ChromaDB (local dev), BigQuery VECTOR_SEARCH (prod) |
| Config | pydantic-settings + python-dotenv |
| Tests | pytest + pytest-asyncio |
| Runtime | Python 3.11, Cloud Run (prod) |

## Repo layout (key paths)

```
app/
  main.py          # FastAPI app entry point
  config.py        # pydantic-settings for all env vars
  models/
    schemas.py     # ALL Pydantic I/O models — source of truth for types
  llm/
    base.py        # LLMClient ABC
    claude_client.py
    gemini_client.py
  retrieval/
    vector_store.py  # VectorStore ABC
    local_store.py   # ChromaDB impl
    bigquery_store.py  # prod stub (NotImplementedError + detailed docstring)
    embedder.py
    ingest.py
  agents/
    intent_agent.py   # classify query → IntentResult
    clarification.py  # generate ClarifyingQuestion
    router.py         # decide action branch
  ranking/
    loyalty_ranker.py  # loyalty-aware re-ranking
  api/
    routes.py       # POST /assist, GET /health
data/
  generate_catalogs.py  # generates catalogs via Claude Haiku
  catalogs/            # generated JSON files (gitignored)
  user_profiles.json   # mock user profiles for testing
tests/
scripts/
  deploy_gcp.sh
  load_test.py
  cost_analysis.py
docs/
  decisions.md   # ADRs live here
```

## Implementation sequence

1. Schemas & validation (schemas.py is complete) — write schema tests first.
2. Catalog generation — run `python data/generate_catalogs.py` once.
3. Retrieval — implement Embedder → LocalChromaStore → ingest.py.
4. LLM clients — implement ClaudeClient, then GeminiClient.
5. Agents — implement intent_agent, clarification, router.
6. Ranking — implement LoyaltyRanker.
7. API — wire everything into routes.py, replace stub.
8. Tests — expand placeholder tests at each step.
9. Docker — verify Dockerfile builds and container starts.

## Critical conventions

- **Pydantic v2 only.** Never use `dict` for I/O; always use a schema from `schemas.py`.
- **Async throughout.** All I/O methods (LLM, vector store, embedder) must be `async`.
- **Single source of truth for models.** Add new fields to `schemas.py` first, then update callers.
- **Both LLM providers must stay in sync.** Any prompt change must be reflected in both
  `claude_client.py` and `gemini_client.py`.
- **Config via env only.** Never hardcode API keys, model names, or file paths — use `settings`.

## Key invariants

- `Partner` enum values must match the keys used in `user_profiles.json` and catalog filenames.
- `LoyaltyRanker` weights must sum awareness: α=0.6, β=0.3, γ=0.1 are defaults — expose them
  as constructor args so tests can override.
- `generate_catalogs.py` is idempotent: it skips partners whose JSON already exists.

## Local dev commands

```bash
# Start API with hot reload
uvicorn app.main:app --reload

# Run all tests
pytest tests/ -v

# Generate catalogs (needs ANTHROPIC_API_KEY)
python data/generate_catalogs.py

# Cost estimate
python scripts/cost_analysis.py --requests-per-day 50000 --provider claude

# Load test (API must be running)
python scripts/load_test.py --rps 20 --duration 60
```

## Environment variables (see .env.example)

All loaded via `app/config.py` (pydantic-settings). Key vars:
- `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` — LLM keys
- `DEFAULT_LLM_PROVIDER` — "claude" or "gemini"
- `VECTOR_STORE` — "local" (ChromaDB) or "bigquery"
- `CHROMA_PERSIST_DIR` — path for ChromaDB persistence (default `./chroma_db`)
- `GCP_PROJECT_ID` / `BIGQUERY_DATASET` — only needed when `VECTOR_STORE=bigquery`
