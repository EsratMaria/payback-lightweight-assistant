# E2E Eval (LLM-as-Judge) — 20260514_162653

**Judge model:** claude-opus-4-7  
**Production model:** claude-sonnet-4-6  
**Queries:** 10  
**Runs per query:** 3  
**Duration:** 506.7s  

## Mean Scores (out of 3)

| Dimension | Mean Score |
|---|---|
| Query Satisfaction | 2.40 |
| Result Quality | 2.10 |
| Language & Tone Match | 2.90 |
| **Total (out of 9)** | **7.40** |

## Score Distribution

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Query Satisfaction | 1 | 0 | 3 | 6 |
| Result Quality | 1 | 1 | 4 | 4 |
| Language Tone | 0 | 0 | 1 | 9 |

## Judge–Human Agreement

Agreement = fraction of queries where |judge_score − human_score| ≤ 1

| Dimension | Agreement |
|---|---|
| query_satisfaction | 88.9% |
| result_quality | 77.8% |
| language_tone | 100.0% |
| overall | 88.9% |

## Per-Query Detail

| Query | Type | QS | RQ | LT | Total | Variance |
|---|---|---|---|---|---|---|
| `stuff for a pasta dinner` | recommendations | 3 | 3 | 3 | 9 | ✓ |
| `ingredients for tiramisu` | recommendations | 0 | 0 | 2 | 2 | ✓ |
| `wireless mouse` | recommendations | 3 | 1 | 3 | 7 | ✓ |
| `Products for glowing skin from dm` | recommendations | 2 | 2 | 3 | 7 | ✓ |
| `something for my dog` | clarification | 3 | 3 | 3 | 9 | ✓ |
| `etwas Schönes für meinen Mann` | clarification | 3 | 2 | 3 | 8 | ✓ |
| `alles für ein Wochenende am See` | recommendations | 2 | 2 | 3 | 7 | ✓ |
| `open Amazon for me` | navigation | 3 | 3 | 3 | 9 | ✓ |
| `take me to the shop` | clarification | 3 | 2 | 3 | 8 | ✓ |
| `how do I redeem my points` | clarification | 2 | 3 | 3 | 8 | ✓ |

## Interpretation

Mean total score is 7.40/9 across 10 queries. Query satisfaction (2.40) measures response-type correctness; result quality (2.10) measures content appropriateness; language tone match (2.90) measures bilingual consistency. No high-variance queries — judge agreement was consistent across runs.

---
