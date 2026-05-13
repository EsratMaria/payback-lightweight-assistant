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
  deploy.sh          # Cloud Run deployment (see scripts/deploy.README.md)
  deploy.README.md   # deploy prerequisites, rotation, teardown, cost
  load_test.py
  cost_analysis.py
docs/
  decisions.md   # ADRs live here
```

## Implementation sequence

All steps complete.

1. ✅ Schemas & validation — `schemas.py`; all Pydantic v2 models including `prefers_deals` and `promo_fallback`.
2. ✅ Catalog generation — ~700 products across dm, EDEKA, Amazon via `data/generate_catalogs.py` (Claude Haiku, tool-use forced output, 80% acceptance threshold, exponential backoff).
3. ✅ Retrieval — `Embedder` (lazy-load, `lru_cache` singleton) → `LocalChromaStore` (ChromaDB cosine, upsert idempotency, `1.0 − distance` clamped, `partner_filter` + `promo_only` where-clause) → `BigQueryVectorStore` stub → `ingest.py`.
4. ✅ LLM clients — `ClaudeClient` (tool-use, Pydantic retry, warning logger); `LLMClient` ABC for provider extensibility.
5. ✅ Intent agent — `IntentAgent` classifies 7 fields: `language`, `intent`, `specificity`, `confidence`, `extracted_query`, `target_partner`, `is_basket_query`, `prefers_deals`.
6. ✅ Router — 5-branch orchestrator: navigational+partner, support, navigational-no-partner, specific+high-confidence (with basket expansion + promo filter), vague/low-confidence. Returns `latency_ms`, `estimated_cost_eur`.
7. ✅ Clarification agent — `ClarificationAgent` generates catalog-grounded clarifying questions.
8. ✅ Query expander — `QueryExpander` does pre-flight retrieval for category discovery, then LLM generates 3-5 sub-queries constrained to available categories; per-sub relevance filter (threshold=0.45); `debug_dropped_queries` surfaced in response.
9. ✅ Loyalty ranker — `LoyaltyRanker` with configurable α=0.6/β=0.3/γ=0.1; cold-start diversity from result-set partner distribution.
10. ✅ Deal-seeking filter — `prefers_deals` intent signal; `promo_only=True` passed to vector store; graceful fallback with `promo_fallback=True` in response.
11. ✅ API — `POST /assist`, `GET /health`, `GET /users`, `GET /users/{id}`, `GET /partners`, `GET /stats`; Swagger UI at `/docs`.
12. ✅ Evaluation infrastructure — `evals/run_intent_eval.py` (accuracy + confusion matrix), `evals/run_retrieval_eval.py` (P@5, R@5), `evals/run_e2e_eval.py` (Opus judge, 3-run median, judge-human agreement).
13. ✅ Tests — 57 unit tests across retrieval, router, ranker, intent, query expander, API, and eval math.
14. ✅ Demo notebook — `notebooks/demo.ipynb` (20 cells, 7 query types, live output, verdict section).
15. ✅ Docker — multi-stage Dockerfile (builder + runtime); CPU-only torch via `--index-url https://download.pytorch.org/whl/cpu`; non-root `appuser`; HEALTHCHECK; `${PORT:-8080}`.
16. ✅ Cloud Run deployment — `scripts/deploy.sh` (Secret Manager setup, API enablement, Cloud Build deploy, health check); `scripts/deploy.README.md` (prerequisites, rotation, teardown, cost).
17. ✅ ADRs — 17 Architecture Decision Records in `docs/decisions.md`.

## Critical conventions

- **Pydantic v2 only.** Never use `dict` for I/O; always use a schema from `schemas.py`.
- **Async throughout.** All I/O methods (LLM, vector store, embedder) must be `async`.
- **Single source of truth for models.** Add new fields to `schemas.py` first, then update callers.
- **Single LLM provider (Claude).** All prompts live in the agent files. Adding a new provider means a new class implementing `LLMClient` — no other file changes.
- **Config via env only.** Never hardcode API keys, model names, or file paths — use `settings`.
- **tool-use for structured LLM output.** Always use `tool_choice={"type":"tool","name":"respond"}` — never ask Claude to "reply in JSON". This eliminates markdown wrapping and JSONDecodeError at extraction.
- **target_partner semantics.** Set whenever a recognized partner (dm, edeka, amazon) is named, regardless of specificity. Null if no partner, multiple partners, or an unsupported partner (REWE, Lidl, etc.) is mentioned. The router, not the agent, decides what to do with it.
- **prefers_deals semantics.** Set by the intent agent on explicit deal signals ("günstig", "Angebot", "cheap", "sale"). The router passes `promo_only=True` to the vector store; if that yields no results, it retries unfiltered and sets `promo_fallback=True` in the response. Never hard-fail on empty promo results — always fall back.
- **promo_only filter.** Implemented as a ChromaDB `where` clause on `active_promo=True`, combined with `partner_filter` via `$and` when both are set. BigQuery stub accepts the parameter but raises `NotImplementedError`.

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

# End-to-end router sanity check (runs 3 deal-seeking queries by default)
python -m scripts.try_router

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

## Deployment

The service is packaged as a Docker image and deployed to GCP Cloud Run.

**Build prerequisites** (must be done before `docker build` or `./scripts/deploy.sh`):
1. Run `python -m app.retrieval.ingest --rebuild` to populate `chroma_db/` — the vector
   index is baked into the image at build time.

**Local Docker test:**
```bash
docker build -t payback-assistant:test .
docker run --rm -p 8080:8080 \
  -e UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC=https://... \
  -e UNIFIED_ENDPOINT_KEY=sk-ant-... \
  payback-assistant:test
curl http://localhost:8080/health
```

**Deploy to Cloud Run:**
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

The script handles Secret Manager setup, API enablement, Cloud Build–based image
construction, and service deployment. See `scripts/deploy.README.md` for the full
guide (prerequisites, key rotation, teardown, cost expectations).
