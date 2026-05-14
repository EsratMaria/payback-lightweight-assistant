# Intent Classification Eval — 20260514_144702

**Model:** claude-sonnet-4-6  
**Dataset:** 25 queries  
**Duration:** 107.1s  

## Per-Dimension Accuracy

| Dimension | Accuracy |
|---|---|
| Language | 100.0% |
| Intent | 100.0% |
| Specificity | 100.0% |
| Is Basket Query | 100.0% |
| Target Partner | 92.0% |
| **Overall (all correct)** | **92.0%** |

## Failures

2 queries failed (partial or complete mismatch):

- **`Zutaten für Lasagne`** — wrong fields: `target_partner`
  - expected: `{'language': 'de', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': True, 'target_partner': None}`
  - actual:   `{'language': 'de', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': True, 'target_partner': 'edeka'}`
  - notes: basket German recipe

- **`Morgenroutine Pflege`** — wrong fields: `target_partner`
  - expected: `{'language': 'de', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': True, 'target_partner': None}`
  - actual:   `{'language': 'de', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': True, 'target_partner': 'dm'}`
  - notes: basket German, beauty regimen

## Interpretation

Overall accuracy is 92.0% across 25 queries. The weakest dimension is **target_partner** at 92.0%. There are 2 failing queries. Review the failures section above.
