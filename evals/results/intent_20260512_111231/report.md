# Intent Classification Eval — 20260512_111231

**Model:** claude-sonnet-4-6  
**Dataset:** 25 queries  
**Duration:** 99.6s  

## Per-Dimension Accuracy

| Dimension | Accuracy |
|---|---|
| Language | 100.0% |
| Intent | 96.0% |
| Specificity | 96.0% |
| Is Basket Query | 100.0% |
| Target Partner | 88.0% |
| **Overall (all correct)** | **80.0%** |

## Failures

5 queries failed (partial or complete mismatch):

- **`wireless mouse`** — wrong fields: `target_partner`
  - expected: `{'language': 'en', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': False, 'target_partner': None}`
  - actual:   `{'language': 'en', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': False, 'target_partner': 'amazon'}`
  - notes: single item English, no partner

- **`USB-C charger`** — wrong fields: `target_partner`
  - expected: `{'language': 'en', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': False, 'target_partner': None}`
  - actual:   `{'language': 'en', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': False, 'target_partner': 'amazon'}`
  - notes: single item English tech, no partner

- **`Zutaten für Lasagne`** — wrong fields: `target_partner`
  - expected: `{'language': 'de', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': True, 'target_partner': None}`
  - actual:   `{'language': 'de', 'intent': 'search', 'specificity': 'specific', 'is_basket_query': True, 'target_partner': 'edeka'}`
  - notes: basket German recipe

- **`take me to the shop`** — wrong fields: `intent`
  - expected: `{'language': 'en', 'intent': 'search', 'specificity': 'navigational', 'is_basket_query': False, 'target_partner': None}`
  - actual:   `{'language': 'en', 'intent': 'discovery', 'specificity': 'navigational', 'is_basket_query': False, 'target_partner': None}`
  - notes: navigational without partner

- **`how do I redeem my points`** — wrong fields: `specificity`
  - expected: `{'language': 'en', 'intent': 'support', 'specificity': 'specific', 'is_basket_query': False, 'target_partner': None}`
  - actual:   `{'language': 'en', 'intent': 'support', 'specificity': 'vague', 'is_basket_query': False, 'target_partner': None}`
  - notes: support English, out-of-scope

## Interpretation

Overall accuracy is 80.0% across 25 queries. The weakest dimension is **target_partner** at 88.0%. There are 5 failing queries. Review the failures section above — clusters in a single field often indicate a prompt gap.
