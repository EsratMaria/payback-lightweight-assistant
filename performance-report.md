# Performance Report

Latency is shaped by three levers: how many LLM calls per request, the size of those calls, and how aggressively the system can short-circuit. The system makes between 1 and 2 LLM calls per request depending on the routing branch, and the routing logic is designed to keep that number as low as possible while preserving quality.

## Routing branches

The router has five branches, three of which short-circuit after a single intent classification with no retrieval:

| Branch | Trigger | LLM calls | Retrieval |
|---|---|---|---|
| A | Navigational + known partner | 1 (intent) | None |
| A.5 | Support intent | 1 (intent) | None |
| A.6 | Navigational + no partner | 1 (intent) | None |
| B | Specific + high confidence | 1–2 (intent + optional expansion) | 1 or N sub-queries |
| C | Vague / low confidence | 2 (intent + clarification) | 1 |

Branches A, A.5, and A.6 return hardcoded responses after a single intent call — no retrieval, no second LLM round-trip. This makes them the fastest paths in the system.

Branch B splits further: single-item queries use one intent call and one retrieval pass. Basket queries (flagged via `is_basket_query`) add a second LLM call for query expansion, followed by 3–5 independent sub-query retrievals with a per-sub relevance filter (threshold 0.45).

## Latency

Single-item specific queries (Branch B, non-basket) complete in roughly **4–6 seconds** on a warm instance, with the intent LLM call dominating. ChromaDB retrieval over the local ~700-product index adds around 50 ms.

Basket queries are the most expensive path — intent + expansion + per-sub-query retrieval + ranking — landing around **6–10 seconds**, dominated by the two LLM round-trips. Retrieval across 3–5 sub-queries still adds only ~150–250 ms total.

The first request after startup incurs a one-time embedding model load (~10–15 seconds); all subsequent requests use the cached singleton.

## Prompt engineering and reliability

All LLM calls use Anthropic tool-use with forced structured output (`tool_choice={"type":"tool","name":"respond"}`) and Pydantic-validated responses, which eliminates the class of retries caused by malformed JSON. On the rare validation failure, the retry loop fires once before raising. System prompts are pre-templated — no per-request reconstruction overhead.

## Cost per request

| Path | LLM calls | Est. cost |
|---|---|---|
| Short-circuit (A, A.5, A.6) | 1 | ~€0.003 |
| Single-item search (Branch B) | 1 | ~€0.003 |
| Basket search (Branch B + expansion) | 2 | ~€0.005 |
| Vague / clarification (Branch C) | 2 | ~€0.005 |

Estimates based on Claude Sonnet input/output token pricing.
