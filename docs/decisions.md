# Architecture Decision Records — PAYBACK Lightweight Assistant

This document captures the meaningful engineering decisions made during the build,
in the order they came up. Each ADR explains *why* a choice was made, not just *what*
was decided. Where decisions came from observed system behavior, the observation is
named explicitly so the rationale is reproducible.

## Index

| # | Title | One-line summary |
|---|---|---|
| ADR-001 | Single-provider Claude with extensibility-by-interface | Build with Claude only; hide it behind an abstract interface so adding a second provider is one new class |
| ADR-002 | Synthetic catalog generation — batched tool-use with per-item validation | 25-product batches, 80% acceptance threshold, Pydantic validation per item |
| ADR-003 | Multilingual embedding model + cross-lingual regression test | MiniLM-L12-v2 with an enforced test asserting EN query retrieves DE product |
| ADR-004 | ChromaDB local store, BigQuery production stub | Working ChromaDB for dev; documented BigQuery stub behind the same interface |
| ADR-005 | target_partner set on any recognized partner mention | Set regardless of specificity; router decides what to do with it |
| ADR-006 | Unknown partners → null target_partner, graceful fallthrough | REWE/Lidl/Aldi → null, search spans available partners instead of erroring |
| ADR-007 | Five-branch router with explicit fast-paths | Navigation, support, partner-picker, retrieve+rank, clarification — each independently tested |
| ADR-008 | Synthetic catalog coverage is intentionally narrow | Documented the gap; decided to not regenerate to make demos look better |
| ADR-009 | Catalog-grounded query expansion — pre-flight + relevance filter | Pre-flight retrieval discovers real categories; per-sub-query floor drops hallucinated sub-queries |
| ADR-010 | Multilingual embedding limits on abstract compound German nouns | Documented the limit; did not lower threshold for German queries |
| ADR-011 | Loyalty ranker — three-signal weighted formula with cold-start fallback | α·semantic + β·commercial + γ·diversity; cold-start uses result-set distribution |
| ADR-012 | Query expansion gated on is_basket_query LLM signal | Boolean from intent agent, not keyword heuristics |
| ADR-013 | LLM-as-judge — Opus judge, three-run median, judge-vs-human agreement | Different model size, variance control, calibrated against human labels |
| ADR-014 | Support intent handled by hardcoded out-of-scope response | No second LLM call; honest scope boundary beats fake helpfulness |
| ADR-015 | Confidence threshold separate from specificity for routing | specificity=specific AND confidence≥0.6 required to reach search branch |
| ADR-016 | BigQuery vector store implementation deferred | No live demo; stub captures design without overclaiming expertise |
| ADR-017 | Sonnet for production, Opus for judge | Model-size variation as bias-mitigation within single-provider constraint |

---

## ADR-001: Single-provider Claude with extensibility-by-interface

**Status:** Accepted

**Context:**
We needed an LLM provider for intent classification, clarification, and query expansion. PAYBACK most probably runs on GCP and uses Gemini in production; the case study references both Claude and Gemini as plausible choices.

**Decision:**
Implement the system with Claude as the single LLM provider, behind an abstract `LLMClient` interface (`app/llm/base.py`). All callers depend on the interface, never on `ClaudeClient` directly.

**Rationale:**
A second LLM implementation would have consumed 2+ hours of SDK setup and provider-specific output-format testing without unlocking new system behavior. The architectural signal that matters is the *interface*: `IntentAgent`, the router, the expander, and the clarification agent all accept any `LLMClient`. Adding a provider is one new class, no caller changes.

**Consequences:**
**I built the interface and one implementation; a second is one new class away.** The `judge_client.py` Sonnet/Opus split (ADR-017) is a degenerate use of the same pattern within one provider. Cross-provider judge eval would further reduce bias (ADR-013).

---

## ADR-002: Synthetic catalog generation — batched tool-use with per-item validation and an acceptance threshold

**Status:** Accepted

**Context:**
We needed ~700 realistic products across three partners (dm, EDEKA, Amazon) including German and English names, realistic categories, and loyalty fields (`points_multiplier`, `active_promo`) baked in from the start.

**Decision:**
Generate catalogs in batches of 25 products per LLM call, using Anthropic tool-use to force structured output, Pydantic-validating each item individually, and accepting batches that meet an 80% validity threshold (≥21/25 valid). Batches below 10/25 valid items are discarded and retried with exponential backoff.

**Rationale:**
One API call per batch amortizes prompt overhead vs. per-item calls. Tool-use eliminates JSON-parse failures at the API level. Per-item Pydantic validation handles individual malformation gracefully — a batch with two bad products still contributes 23 good ones. The acceptance threshold sets an explicit quality floor rather than accepting whatever the LLM produced on a bad run.

**Consequences:**
~700 products generated in roughly two minutes at marginal API cost. Catalogs are idempotent — `generate_catalogs.py` skips partners whose JSON already exists. The same batched-tool-use + Pydantic-validation pattern is used for all structured LLM outputs in the system.

---

## ADR-003: Multilingual embedding model — paraphrase-multilingual-MiniLM-L12-v2 + cross-lingual regression test

**Status:** Accepted

**Context:**
PAYBACK serves German users primarily. The system must retrieve relevantly when queries and product descriptions span English and German — a German user searching "Windeln" must retrieve dm's "Pampers" and "babylove" products without any translation step.

**Decision:**
Use `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~50 languages). Enforce cross-lingual capability with a regression test: `test_cross_lingual_retrieval` asserts that the English query "milk" surfaces a German EDEKA product ("EDEKA Bio Vollmilch") in the top-5 results.

**Rationale:**
This model is widely benchmarked for German, runs locally with no GCP dependency during development, loads in under 8 seconds, and is small enough for fast iteration. The 768-dim alternative (`paraphrase-multilingual-mpnet-base-v2`) was considered but deferred until limits surfaced in practice. The regression test turns a capability claim into a verifiable assertion — cross-lingual retrieval is enforced, not hoped-for. Limits discovered later are documented in ADR-010.

**Consequences:**
Cross-lingual retrieval works for concrete nouns. Abstract compound German nouns surface the model's limits (ADR-010). The model is swappable via the `Embedder` class without changing retrieval or ingest logic.

---

## ADR-004: ChromaDB local store, BigQuery production stub

**Status:** Accepted

**Context:**
We need a vector store for ~700 products in development, and a clear path to the production backend. The case study suggests a BigQuery footprint; co-locating embeddings there is the natural production architecture.

**Decision:**
Implement `LocalChromaStore` (working, persistent, cosine similarity, used throughout the demo and all tests). Implement `BigQueryVectorStore` as a stub with a module-level docstring covering production schema, the `VECTOR_SEARCH` query pattern, `ML.GENERATE_EMBEDDING` integration, and migration path. Both share the abstract `VectorStore` interface.

**Rationale:**
ChromaDB is behaviorally identical to any vector store at ~700 products, runs locally with no GCP cost, and enables fast iteration. The stub is a legitimate deliverable — it captures the production design in code (the docstring) without overclaiming hands-on expertise not yet held. The two implementations prove the interface is real: swapping is a single line in `get_default_router()`. See ADR-016 for the explicit deferral rationale.

**Consequences:**
Dev and demo ship with ChromaDB. Production migration is a configuration change (`VECTOR_STORE=bigquery`) plus a backfill ingest — no code refactor. The stub's schema and query pattern are sufficient for a production engineer to implement without starting from scratch. more one why this decision was made ---> `scripts/deploy.README.md`

---

## ADR-005: target_partner set whenever a recognized partner is named, regardless of intent specificity

**Status:** Accepted

**Context:**
Initial behavior: `target_partner` was set only when `specificity=navigational`. Observed during testing: "pasta from edeka" classified as `specificity=specific` left `target_partner=null`, dropping the useful partner-constraint signal that retrieval can use as a filter.

**Decision:**
`target_partner` is set whenever the user explicitly names a recognized partner (dm, EDEKA, Amazon), regardless of whether the intent is navigational or search-specific. The router then decides what to do with it based on `specificity`.

**Rationale:**
One signal, one source of truth. Adding a separate `partner_constraint` field alongside `target_partner` would be schema bloat — the same information expressed twice with different names. The intent agent's job is to surface that a partner was named; routing decisions belong in the router. This avoids the agent making routing decisions implicitly by sometimes setting and sometimes not setting a field.

**Consequences:**
The `specificity=specific + target_partner=set` router branch applies `partner_filter` to retrieval, narrowing results to that partner's catalog. The `specificity=navigational + target_partner=set` branch returns a navigation response. Same field, two distinct router behaviors — cleanly separated.

---

## ADR-006: Unknown partners (REWE, Lidl, Aldi) → null target_partner, graceful fallthrough

**Status:** Accepted

**Context:**
Users may mention PAYBACK partners we don't carry. "Pasta from REWE" should not fail with an error or pretend to filter on a partner the system can't serve.

**Decision:**
When the user names a partner not in our `Partner` enum, the intent agent sets `target_partner=null` and records the rejection in the `reasoning` field. Retrieval falls through to span our three available partners.

**Rationale:**
The `Partner` enum hard-enforces valid values via Pydantic — an invalid value would trigger a validation error and retry. The system prompt guides the LLM to handle this case explicitly: set null, explain in reasoning. The result is a useful cross-partner search rather than an error. The reasoning field makes the fallthrough behavior debuggable without surfacing it to the user.

**Consequences:**
Users mentioning out-of-scope partners get useful results from the catalog. This pattern scales — PAYBACK has hundreds of real partners; the same fallthrough handles any unrecognized name without code changes.

---

## ADR-007: Five-branch router with explicit fast-paths for navigation, support, and partner-picker

**Status:** Accepted

**Context:**
A naive router has two branches: "specific → retrieve" and "everything else → clarification." This conflates navigational intent (user wants to open a partner app), support intent (user has an account question), partner disambiguation (user said "take me to the shop" without specifying which), and genuinely vague queries.

**Decision:**
The router has five branches in priority order: (1) navigational + known partner → navigation response; (2) support intent → hardcoded out-of-scope clarification, no retrieval; (3) navigational without partner → hardcoded partner-picker, no retrieval; (4) specific + high-confidence → retrieve + rank; (5) fall-through → catalog-grounded clarification.

**Rationale:**
Branches 2 and 3 are hardcoded fast-paths: faster, cheaper, and more reliable than letting an LLM generate generic clarifications for these fully-determined cases. Cost efficiency is maintained. Each branch is independently testable with mocked dependencies. The fall-through path (branch 5) retrieves before clarifying so the clarification agent has catalog context to ground its question — not a cold LLM call.

**Consequences:**
Branches 2 and 3 cost one LLM call (intent classification only) and respond in ~3-5s. Branch 4 costs one or two LLM calls depending on basket expansion. Each branch has its own test coverage in `tests/test_router.py`.

---

## ADR-008: Synthetic catalog coverage is intentionally narrow per category

**Status:** Accepted

**Context:**
Observed during testing: "Schokolade" returns chocolate-flavored yogurts and ice creams — no chocolate bars in the catalog. "Stuff for pasta dinner" returns pasta, and spaghetti from EDEKA only, because dm and Amazon don't stock much grocery items in the synthetic data. Pre-flight category discovery surfaces these gaps honestly.

**Decision:**
Kept the catalog as-is. Documented the limit. Did not regenerate to make demo queries look better.

**Rationale:**
The system is doing the correct thing on what's there: retrieval ranks by semantic similarity, the ranker mixes partners when partners overlap, dropped sub-queries are surfaced transparently. Regenerating to fill gaps would (a) churn the catalogs and require re-running all evals, (b) feel like cherry-picking to flatter the demo, (c) not change the system behavior. The honest fix for production is real PAYBACK partner catalogs replacing the synthetic data — no code changes needed.

**Consequences:**
Demo queries for sparse categories return less satisfying results, and that is a documented expected outcome with a clear production fix. The ranker's diversity signal is observable when partners overlap; the expansion's dropped-query transparency shows catalog gaps without hiding them.

---

## ADR-009: Catalog-grounded query expansion — pre-flight category discovery + per-sub-query relevance filter

**Status:** Accepted

**Context:**
Observed during testing of "stuffs for glam makeup": the LLM expander proposed "glitter eyeshadow palette" and "setting spray" based on real-world brand knowledge — but those categories don't exist in the synthetic catalog. The prompt said "stay grounded in what our partners sell" but the LLM had no way to know catalog contents.

**Decision:**
Two-pass expansion. First, a pre-flight retrieval (no LLM call) on the raw query discovers which categories the catalog actually has. That sorted, deduplicated category list is injected into the expansion prompt. Second, after the LLM proposes sub-queries, each one goes through retrieval with a configurable similarity floor (default 0.45, `EXPANSION_MIN_RELEVANCE`). Sub-queries returning nothing above the floor are dropped before merging. Both proposed and dropped sub-queries are surfaced in `debug_expanded_queries` / `debug_dropped_queries`.

**Rationale:**
Prompt-grounding failed because the LLM has no access to live catalog state. Data-grounding works because the pre-flight tells it exactly what categories are stocked. The per-sub-query filter is a safety net for sub-queries that sound plausible but retrieve nothing useful.

**Consequences:**
Pre-flight adds ~50ms (a ChromaDB query). The dropped-queries list is honest signal about catalog gaps. I observed the reality of not matching design and fixed it with grounding, not prompt tuning.

---

## ADR-010: Multilingual embedding limits on abstract compound German nouns

**Status:** Accepted

**Context:**
Observed during testing: "essentials for moving into a new house" produces a sensible mixed-partner basket. The German equivalent "Grundausstattung für eine neue Wohnung" surfaces bakery items and chicken breast. Pre-flight retrieval returns weakly-matched adjacent-category items; expansion sub-queries like `Haushaltswaren` and `Heimtextilien` retrieve nothing above the 0.45 threshold.

**Decision:**
Documented the limit. Did not lower the relevance threshold for German queries.

**Rationale:**
The symptom is an embedding model limitation, not a threshold problem. `paraphrase-multilingual-MiniLM-L12-v2` handles simple, concrete bilingual terms well but is weak on abstract compound nouns — a known characteristic of 384-dim multilingual models. Lowering the threshold for German would have admitted *worse matches*, producing a larger basket polluted by weak items rather than a smaller basket of correct ones. The honest fix is a stronger model (`paraphrase-multilingual-mpnet-base-v2` or Vertex AI's `text-multilingual-embedding-002`), swappable via the `Embedder` interface.

**Consequences:**
Simple German queries ("Windeln", "Bio Tomaten") work well. Abstract compound-noun queries surface the limit honestly. Production fix is a model swap, not a code refactor. The ADR is the documentation that this is known and understood, not an oversight.

---

## ADR-011: Loyalty ranker — three-signal weighted formula with cold-start fallback

**Status:** Accepted

**Context:**
PAYBACK's business model rewards users diversifying across partners. Pure semantic similarity would always surface the partner with the most semantically-matching items in the synthetic catalog, defeating the loyalty diversification goal.

**Decision:**
Implemented the ranker as a three-signal weighted sum: `final_score = α·semantic + β·commercial + γ·diversity` with defaults α=0.6, β=0.3, γ=0.1 (all configurable via config). Commercial boost (β) combines `points_multiplier` and `active_promo`. Diversity bonus (γ) has two modes: known user (`(1 - affinity[partner]) * scale`) and cold-start (`1 - (partner_count_in_topk / topk)`).

**Rationale:**
Three signals, each independently defensible and bounded to [0, 1], so the composite score is also bounded. The cold-start path uses the same equation shape with a different input — result-set partner distribution — avoiding special-case branching logic. Weights are read from `config` on each call. Alternative considered: a learning-to-rank model. Rejected as over-engineering for a 700-product catalog with synthetic user data.

**Consequences:**
Loyalty signal is observable in mixed-partner result sets (Query 3 vs Query 4 in the demo notebook). For single-partner-dominated queries, the diversity bonus is small — that's the correct behavior, not a bug.

---

## ADR-012: Query expansion gated on is_basket_query LLM signal, not keyword heuristics

**Status:** Accepted

**Context:**
Query expansion adds latency (~1-2s extra for LLM call + per-sub-query retrievals) and cost. However It fires only when a query implies multiple related products worth expanding. The question was how to detect that.

**Decision:**
Add a boolean `is_basket_query` field to the intent agent's structured `IntentResult` output. The intent agent decides — as part of its existing classification call — whether the query implies a basket of related items. The router uses this signal exclusively to gate expansion.

**Rationale:**
Keyword heuristics ("contains 'dinner', 'party', 'stuff for'") would be brittle across German phrasing and miss legitimate compound intents in unexpected formulations. The intent agent is already reasoning semantically about the query; one additional boolean costs nothing in latency or tokens (it's part of the same structured output). LLM-as-classifier outperforms regex-as-classifier here with no added cost.

**Consequences:**
Expansion fires when it should ("pasta dinner" → `is_basket_query=True`) and skips when it shouldn't ("wireless mouse" → `is_basket_query=False`). Observable in `debug_expanded_queries` in the response. The signal can be overridden per-request in tests via mock intent results.

---

## ADR-013: LLM-as-judge evaluation — Opus judge, three-run median, judge-vs-human agreement

**Status:** Accepted

**Context:**
We needed an end-to-end relevance metric. Standard LLM-as-judge has two known failure modes: stochastic variance (scores change run-to-run) and self-preference bias (the same model that generated a response tends to rate it more favourably).

**Decision:**
Use Claude Opus as judge while the production system uses Sonnet (ADR-017). Score three dimensions — `query_satisfaction`, `result_quality`, `language_tone_match` — on a 0-3 ordinal scale. Run each query three times and take the median per dimension. Flag queries where the score range across runs exceeds 1 as high-variance. Track judge-vs-human agreement on a hand-rated 10-query validation set; without calibration, the metric is unfounded.

**Rationale:**
Variance control on stochastic LLM output is the difference between a metric and a number. Three runs with median + variance flagging is the lightest credible setup. Ordinal 0-3 is easier to calibrate human labels against than continuous [0,1]. Judge-vs-human agreement validates the judge before trusting its scores at scale.

**Consequences:**
Three Opus runs per query costs more per eval run, but eval runs are infrequent. The Sonnet/Opus split is single-provider only; cross-provider judging (e.g. GPT-4) would further reduce bias and is supported by the `LLMClient` interface.

---

## ADR-014: Support intent handled by hardcoded out-of-scope response, not LLM clarification

**Status:** Accepted

**Context:**
Queries like "how do I redeem my points" classify as `intent=support` with high confidence. Initially, these fell through to the clarification path, where the LLM generated plausible-but-invented options like "redeem at EDEKA / dm / Amazon" — factually wrong and misleading.

**Decision:**
When `intent == support`, short-circuit to a hardcoded multilingual out-of-scope response pointing to the PAYBACK support contact page. No retrieval, no second LLM call. The message is different in German and English.

**Rationale:**
Honest scope boundary over fake helpfulness. The product is a shopping assistant; account and points questions belong in PAYBACK support flows. The hardcoded response is faster (one LLM call total), cheaper, and more reliable than asking an LLM to improvise scope boundaries. A system that knows what it is not is easier to trust.

**Consequences:**
Support queries cost one LLM call (intent classification only) and respond in ~3-5s. 

---

## ADR-015: Confidence threshold separate from specificity for routing to search

**Status:** Accepted

**Context:**
The intent agent reports both `specificity` (categorical: specific/vague/navigational) and `confidence` (continuous: 0.0-1.0). Routing on `specificity` alone misses cases where the LLM marked a query "specific" but with low confidence — signaling it was uncertain about its own classification.

**Decision:**
The router routes to the search branch only when `specificity == specific AND confidence >= 0.6`. Queries where the intent agent flagged specificity as "specific" but confidence is below 0.6 fall through to catalog-grounded clarification. The threshold is read from config (`INTENT_CONFIDENCE_THRESHOLD`, default 0.6).

**Rationale:**
Confidence is meta-signal — the LLM's uncertainty about its own output. Treating it as a routing gate rather than ignoring it traps genuinely borderline queries before they reach retrieval. The threshold is config-driven because the right value depends on production traffic data; 0.6 is a reasonable starting point.

**Consequences:**
Some borderline queries receive a clarifying question instead of potentially irrelevant recommendations. Moving the threshold to 0.7 is more conservative; 0.5 is more permissive. Observed in practice: `specificity=specific, confidence=0.3` correctly falls to clarification rather than attempting retrieval.

---

## ADR-016: BigQuery vector store implementation deferred

**Status:** Accepted

**Context:**
The case study suggests BigQuery as the production vector backend. PAYBACK operates a large BigQuery footprint; co-locating embeddings there avoids a separate vector database and enables native joins with purchase history and user signal data.

**Decision:**
Deliver `LocalChromaStore` as the working implementation and `BigQueryVectorStore` as a documented stub. The stub's module docstring covers: production table schema, vector index DDL, the `VECTOR_SEARCH` + `ML.GENERATE_EMBEDDING` query pattern, and the migration path from ChromaDB.

**Rationale:**
Two practical constraints drove the deferral. First, hands-on production experience with BigQuery's `VECTOR_SEARCH` function and `ML.GENERATE_EMBEDDING` workflow was not held at the time of implementation — learning a new backend deeply enough to ship production-grade code, within the one-week project scope, was not realistic. Second, the architectural signal that matters most — *"the system is designed to swap backends via an interface"* — is fully captured by both classes existing behind the same `VectorStore` ABC. A rigorously documented stub demonstrates the design without overclaiming expertise.

**Consequences:**
The demo ships with ChromaDB. Production migration is a configuration change (`VECTOR_STORE=bigquery`) plus a backfill ingest — no code refactor. So my honest answer: *"I have not deployed BigQuery vector search in production. I tried designing the stub with claude's help so that its somewhat infrastructure-ready"*

---

## ADR-017: Sonnet for production, Opus for judge — bias-mitigation within single-provider constraint

**Status:** Accepted

**Context:**
With a single LLM provider (ADR-001), cross-provider judging is not available. Same-model judging — using Sonnet to evaluate Sonnet's own outputs — risks self-preference bias: the model may rate its own style and outputs more favourably than an independent evaluator would.

**Decision:**
The system (intent agent, clarification agent, query expander) uses Claude Sonnet (`claude-sonnet-4-6`). The eval judge uses Claude Opus (`claude-opus-4-7`). These are implemented as separate clients: `ClaudeClient` for production, `OpusJudgeClient` for evaluation.

**Rationale:**
Within a single-provider constraint, model-size variation is the strongest available bias-mitigation lever. Opus is Anthropic's most capable model and is less likely to rate Sonnet's outputs favourably due to stylistic similarity. The two-client approach is a degenerate use of the `LLMClient` interface pattern from ADR-001 — different classes, same interface. Cross-provider judging would further reduce bias and is supported by the interface.

**Consequences:**
Eval runs cost more (Opus is priced higher than Sonnet) but eval frequency is low. The limitation — correlated biases within Anthropic's model family — is acknowledged in the `judge_client.py` docstring and in ADR-013.
