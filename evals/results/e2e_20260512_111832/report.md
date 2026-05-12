# E2E Eval (LLM-as-Judge) — 20260512_111832

**Judge model:** claude-opus-4-7  
**Production model:** claude-sonnet-4-6  
**Queries:** 10  
**Runs per query:** 3  
**Duration:** 232.7s  

## Mean Scores (out of 3)

| Dimension | Mean Score |
|---|---|
| Query Satisfaction | 1.80 |
| Result Quality | 1.30 |
| Language & Tone Match | 2.60 |
| **Total (out of 9)** | **5.70** |

## Score Distribution

| Dimension | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Query Satisfaction | 0 | 4 | 4 | 2 |
| Result Quality | 3 | 3 | 2 | 2 |
| Language Tone | 0 | 2 | 0 | 8 |

## Judge–Human Agreement

_No human ratings available yet. Fill in `human_rating_*` fields in_
_`evals/datasets/judge_validation.yaml`, then re-run this eval._

## Per-Query Detail

| Query | Type | QS | RQ | LT | Total | Variance |
|---|---|---|---|---|---|---|
| `stuff for a pasta dinner` | recommendations | 1 | 0 | 1 | 2 | ✓ |
| `ingredients for tiramisu` | recommendations | 1 | 0 | 1 | 2 | ✓ |
| `wireless mouse` | recommendations | 2 | 0 | 3 | 5 | ✓ |
| `Products for glowing skin from dm` | recommendations | 1 | 1 | 3 | 5 | ✓ |
| `something for my dog` | clarification | 3 | 3 | 3 | 9 | ✓ |
| `etwas Schönes für meinen Mann` | clarification | 2 | 1 | 3 | 6 | ✓ |
| `alles für ein Wochenende am See` | recommendations | 1 | 1 | 3 | 5 | ✓ |
| `open Amazon for me` | navigation | 3 | 3 | 3 | 9 | ✓ |
| `take me to the shop` | clarification | 2 | 2 | 3 | 7 | ✓ |
| `how do I redeem my points` | clarification | 2 | 2 | 3 | 7 | ✓ |

## Interpretation

Mean total score is 5.70/9 across 10 queries. Query satisfaction (1.80) measures response-type correctness; result quality (1.30) measures content appropriateness; language tone match (2.60) measures bilingual consistency. No high-variance queries — judge agreement was consistent across runs.

---

<!-- TODO: Once Maria has filled in human_rating_* fields in judge_validation.yaml,
re-run this eval to populate the Judge–Human Agreement section above.
Target: ≥75% within-1-point agreement per dimension. Below 70% means
the judge prompt needs revision. -->