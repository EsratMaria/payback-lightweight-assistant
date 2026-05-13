# PAYBACK Lightweight Assistant

Backend microservice for the PAYBACK loyalty programme's in-app product discovery feature.
Accepts natural-language queries in German or English, classifies intent via Claude,
retrieves relevant products from a multilingual vector store, re-ranks them with a
loyalty-aware scoring function, and returns structured JSON.

---

## Implementation status

| # | What | Status |
|---|---|---|
| 1 | Schemas & Pydantic v2 models | Done |
| 2 | Synthetic catalog generation (~700 products, 3 partners) | Done |
| 3 | Retrieval — multilingual embeddings, ChromaDB, ingest | Done |
| 4 | LLM client — ClaudeClient (tool-use), LLMClient interface | Done |
| 5 | Intent agent — 7-field IntentResult with prefers_deals signal | Done |
| 6 | Router — 5-branch orchestrator with cost + latency metadata | Done |
| 7 | Clarification agent — catalog-grounded clarifying questions | Done |
| 8 | Query expander — catalog-grounded basket expansion with dropped-query metadata | Done |
| 9 | Loyalty ranker — α·semantic + β·commercial + γ·diversity, cold-start fallback | Done |
| 10 | Deal-seeking filter — promo_only hard filter with graceful fallback | Done |
| 11 | API — POST /assist + 5 supporting GET endpoints, Swagger UI | Done |
| 12 | Evaluation infrastructure — intent eval, retrieval P@5/R@5, E2E LLM-as-judge | Done |
| 13 | Tests — 57 unit tests across all layers | Done |
| 14 | Demo notebook — 20 cells, 7 query types, live output | Done |
| 15 | Docker — multi-stage image, CPU-only torch, non-root user | Done |
| 16 | Cloud Run deployment — deploy.sh, Secret Manager, deploy.README.md | Done |
| 17 | Architecture Decision Records — 17 ADRs in docs/decisions.md | Done |

---

## Architecture

```
User query (EN/DE)
       │
       ▼
  IntentAgent  ──────────────────────────────────────────────────────────
  (Claude)                                                              │
  • language, intent, specificity, confidence                          │
  • target_partner, is_basket_query, prefers_deals                     │
       │                                                               │
       ▼                                                               │
    Router  ─────────────────────────────────────────────────────┐    │
  5 branches:                                                     │    │
  A   navigational + partner  → navigation_target                 │    │
  A.5 support                 → out-of-scope message              │    │
  A.6 navigational, no partner→ partner clarification             │    │
  B   specific + confident    → retrieve → rank → recommendations │    │
  C   vague / low-confidence  → retrieve for grounding → clarify  │    │
       │                                                               │
  [Branch B]                                                           │
       │                                                               │
  is_basket? ──yes──▶ QueryExpander (Claude)                          │
       │                pre-flight retrieval → category discovery     │
       │                LLM generates 3-5 sub-queries                 │
       │                per-sub relevance filter (≥0.45)              │
       │                dropped queries surfaced in response           │
       │◀──no──────────────────────────────────────────────────────────
       │
  prefers_deals? ──yes──▶ VectorStore.search(promo_only=True)
       │                   ──empty?──▶ fallback + promo_fallback=True
       │◀──no────────────────────────────────────────────────────────
       │
  LocalChromaStore  (cosine similarity, partner + promo filters)
  paraphrase-multilingual-MiniLM-L12-v2  (384-dim, EN+DE)
       │
  LoyaltyRanker
  final = 0.6·semantic + 0.3·commercial + 0.1·diversity
  cold-start: diversity_bonus from result-set partner distribution
       │
  AssistantResponse (JSON)
```

For full rationale and trade-offs, see [docs/decisions.md](docs/decisions.md) (17 ADRs).

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/assist` | Main endpoint — classify, retrieve, rank, return |
| `GET` | `/health` | Liveness probe |
| `GET` | `/users` | List mock user profiles |
| `GET` | `/users/{user_id}` | Single user profile |
| `GET` | `/partners` | List supported partners |
| `GET` | `/stats` | Catalog item counts per partner |

Interactive docs at `/docs` (Swagger UI) and `/redoc`.

**Sample request:**
```bash
curl -X POST http://localhost:8000/assist \
  -H 'Content-Type: application/json' \
  -d '{"query": "günstige Windeln bei dm", "user_id": "user_edeka_heavy"}'
```

**Sample response fields:**
```json
{
  "response_type": "recommendations",
  "intent_result": {
    "language": "de",
    "intent": "search",
    "specificity": "specific",
    "confidence": 0.97,
    "target_partner": "dm",
    "is_basket_query": false,
    "prefers_deals": true,
    "extracted_query": "Windeln dm günstig",
    "reasoning": "..."
  },
  "recommendations": [...],
  "promo_fallback": false,
  "latency_ms": 4200,
  "estimated_cost_eur": 0.0032
}
```

---

## Key features

### Multilingual retrieval
`paraphrase-multilingual-MiniLM-L12-v2` maps German and English queries to the same
embedding space. An English query for "milk" surfaces the German-language EDEKA product
"Bio Vollmilch". Validated by a cross-lingual regression test in `tests/test_retrieval.py`.

### Catalog-grounded query expansion
For basket queries ("pasta dinner", "Geburtstagsparty"), the router:
1. Runs a pre-flight retrieval to discover available catalog categories
2. Tells the LLM expander exactly which categories exist (prevents hallucinated sub-queries)
3. Searches each sub-query independently; drops sub-queries below a relevance threshold (0.45)
4. Surfaces `debug_expanded_queries` and `debug_dropped_queries` in the response

### Deal-seeking filter
When the query contains deal signals ("günstig", "Angebot", "cheap", "sale"), the router
passes `promo_only=True` to the vector store, restricting results to `active_promo=True`
products. If no promoted products match, the filter is dropped and `promo_fallback=True`
is set in the response.

### Loyalty ranker
```
final_score = α·semantic_score + β·commercial_boost + γ·diversity_bonus
            = 0.6·cosine_sim  + 0.3·(multiplier·promo) + 0.1·partner_diversity
```
Cold-start users (no purchase history) get diversity bonus from the result-set partner
distribution rather than from a user profile.

### Evaluation infrastructure (`evals/`)
- **Intent eval** — accuracy + confusion matrix against a labelled dataset
- **Retrieval eval** — Precision@5 and Recall@5 per query
- **E2E eval** — LLM-as-judge (Claude Opus) scoring query satisfaction, result quality,
  and language/tone match; 3-run median to suppress variance; judge-human agreement scaffold

---

## Quick start

```bash
# 1. Create virtual environment (Python 3.13+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env: set UNIFIED_ENDPOINT_KEY and UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC

# 3. Generate product catalogs (~700 products via Claude)
python data/generate_catalogs.py

# 4. Ingest catalogs into ChromaDB
python -m app.retrieval.ingest --rebuild

# 5. Run tests
pytest tests/ -v -m "not integration"

# 6. Start API
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

---

## Stack

| Layer | Choice | Reason |
|---|---|---|
| API | FastAPI + Pydantic v2 | Async, auto-validation, OpenAPI docs |
| LLM | Anthropic Claude (tool-use) | Forced structured output, no JSON parsing fragility |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | EN+DE in one model, 384-dim, fast |
| Vector store | ChromaDB (local), BigQuery stub (prod) | No infra for dev; BQ joins loyalty data at scale |
| Config | pydantic-settings + python-dotenv | Type-safe env vars |
| Tests | pytest + pytest-asyncio | 57 unit tests |
| Container | Docker multi-stage (CPU torch) | ~511 MB image, non-root, Cloud Run ready |
| Deploy | GCP Cloud Run + Secret Manager | Serverless, scales to zero, credential hygiene |

---

## Testing

```bash
# All unit tests (no real API calls)
pytest tests/ -v -m "not integration"

# Including integration tests (needs UNIFIED_ENDPOINT_KEY)
pytest tests/ -v

# Specific layers
pytest tests/test_retrieval.py tests/test_router.py tests/test_ranker.py -v
```

57 tests cover: retrieval (embedder, ChromaDB, partner filter, promo filter, cross-lingual),
router (all 5 branches, basket expansion, dropped queries, promo filter, promo fallback),
ranker (weights, cold-start, diversity), intent agent (mock LLM), query expander (pre-flight,
categories, empty catalog), API endpoints, and eval metric math.

---

## Deployment

**Prerequisites:** populate `chroma_db/` first — the vector index is baked into the image.

```bash
python -m app.retrieval.ingest --rebuild
```

**Local Docker test:**
```bash
docker build -t payback-assistant:test .
docker run --rm -p 8080:8080 \
  -e UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC=https://... \
  -e UNIFIED_ENDPOINT_KEY=sk-ant-... \
  payback-assistant:test
curl http://localhost:8080/health
```

**Cloud Run:**
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh   # handles Secret Manager, Cloud Build, health check
```

See [scripts/deploy.README.md](scripts/deploy.README.md) for prerequisites, key rotation,
teardown steps, and cost expectations (~€5/month for prototype workloads).

---

## Cost estimates

| Workload | Est. LLM cost |
|---|---|
| 1 000 requests/day | ~€3.20/day |
| 50 000 requests/day | ~€160/day |

```bash
python scripts/cost_analysis.py --requests-per-day 50000 --provider claude
```

Cloud Run infrastructure adds < €5/month at prototype scale (scales to zero between requests).
