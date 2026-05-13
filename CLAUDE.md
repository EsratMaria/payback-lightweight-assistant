# CLAUDE.md — PAYBACK Lightweight Assistant

This file gives Claude Code the context it needs to work effectively in this repo.

## Architecture decisions

The full set of decisions made during the build — with rationale and tradeoffs —
lives in `docs/decisions.md`. New contributors should skim that first.

## Project purpose

Backend microservice for the PAYBACK loyalty app's lightweight assistant feature.
Takes a user query (EN/DE), classifies intent via LLM, routes to the right action
(search, clarify, navigate), returns structured JSON with product recommendations.

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI + Pydantic v2 |
| LLM | Anthropic Claude via `ClaudeClient`; extensible via `LLMClient` interface |
| Embeddings | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) |
| Vector store | ChromaDB with `hnsw:space=cosine` (local dev), BigQuery VECTOR_SEARCH (prod) |
| Config | pydantic-settings + python-dotenv |
| Tests | pytest + pytest-asyncio |
| Runtime | Python 3.13, Cloud Run (prod) |

## Repo layout (key paths)

```
app/
  main.py          # FastAPI app entry point
  config.py        # pydantic-settings for all env vars
  models/
    schemas.py     # ALL Pydantic I/O models — source of truth for types
  llm/
    base.py        # LLMClient ABC — interface for additional providers
    claude_client.py
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

1. ✅ Schemas & validation — `schemas.py` complete; all Pydantic v2 models.
2. ✅ Catalog generation — ~700 products across dm, EDEKA, Amazon via `data/generate_catalogs.py` (Claude Opus 4.5, tool-use forced output, 80% acceptance threshold, exponential backoff).
3. ✅ Retrieval — `Embedder` (lazy-load, `lru_cache` singleton) → `LocalChromaStore` (ChromaDB cosine, upsert idempotency, `1.0 − distance` clamped similarity) → `ingest.py` (returns `{partner: count}` summary, `--rebuild` flag).
4. ✅ LLM clients — `ClaudeClient` (tool-use, Pydantic retry, warning logger) complete; `LLMClient` interface ready for additional providers via dependency injection.
5. ✅ Intent agent — `IntentAgent` + `get_default_intent_agent()` factory complete; `target_partner` set on any recognized partner mention, null on unsupported/multiple. `router.py` and `clarification.py` still pending.
6. ⬜ Ranking — implement `LoyaltyRanker` (α=0.6, β=0.3, γ=0.1).
7. ⬜ API — wire everything into `routes.py`, replace stub.
8. ⬜ Tests — expand for router, clarification, ranker, API.
9. ⬜ Docker — verify Dockerfile builds and container starts.

## Critical conventions

- **Pydantic v2 only.** Never use `dict` for I/O; always use a schema from `schemas.py`.
- **Async throughout.** All I/O methods (LLM, vector store, embedder) must be `async`.
- **Single source of truth for models.** Add new fields to `schemas.py` first, then update callers.
- **Single LLM provider (Claude).** All prompts live in the agent files. Adding a new provider means a new class implementing `LLMClient` — no other file changes.
- **Config via env only.** Never hardcode API keys, model names, or file paths — use `settings`.
- **tool-use for structured LLM output.** Always use `tool_choice={"type":"tool","name":"respond"}` — never ask Claude to "reply in JSON". This eliminates markdown wrapping and JSONDecodeError at extraction.
- **target_partner semantics.** Set whenever a recognized partner (dm, edeka, amazon) is named, regardless of specificity. Null if no partner, multiple partners, or an unsupported partner (REWE, Lidl, etc.) is mentioned. The router, not the agent, decides what to do with it.

## Key invariants

- `Partner` enum values must match the keys used in `user_profiles.json` and catalog filenames.
- `LoyaltyRanker` weights must sum awareness: α=0.6, β=0.3, γ=0.1 are defaults — expose them
  as constructor args so tests can override.
- `generate_catalogs.py` is idempotent: it skips partners whose JSON already exists.
- ChromaDB cosine distance can slightly exceed 1.0 due to floating-point rounding. Always clamp: `similarity = max(0.0, min(1.0, 1.0 - distance))`.
- `ClaudeClient` retries once on `ValidationError` or missing `tool_use` block, then raises `LLMError`. All other exceptions (network, auth, rate-limit) propagate to the caller unchanged.
- Run scripts as modules from repo root: `python -m scripts.try_intent` not `python scripts/try_intent.py`.

## Local dev commands

```bash
# Start API with hot reload
uvicorn app.main:app --reload

# Run all tests
pytest tests/ -v

# Run only unit tests (no real API calls)
pytest tests/ -v -m "not integration"

# Generate catalogs (needs UNIFIED_ENDPOINT_KEY)
python data/generate_catalogs.py

# Ingest catalogs into ChromaDB (re-ingest from scratch)
python -m app.retrieval.ingest --rebuild

# Sanity-check intent classification against real Claude
python -m scripts.try_intent

# Cost estimate
python scripts/cost_analysis.py --requests-per-day 50000 --provider claude

# Load test (API must be running)
python scripts/load_test.py --rps 20 --duration 60
```

## Environment variables (see .env.example)

All loaded via `app/config.py` (pydantic-settings). Key vars:
- `UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC` — base URL for the Anthropic unified endpoint
- `UNIFIED_ENDPOINT_KEY` — API key for the unified endpoint
- `DEFAULT_LLM_PROVIDER` — currently only "claude" is implemented; see `LLMClient` interface for extensibility
- `VECTOR_STORE` — "local" (ChromaDB) or "bigquery"
- `CHROMA_PERSIST_DIR` — path for ChromaDB persistence (default `./chroma_db`)
- `GCP_PROJECT_ID` / `BIGQUERY_DATASET` — only needed when `VECTOR_STORE=bigquery`
