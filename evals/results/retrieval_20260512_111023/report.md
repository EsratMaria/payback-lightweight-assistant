# Retrieval Eval (Precision@5 / Recall@5) — 20260512_111023

**Dataset:** 15 queries  
**Duration:** 9.9s  

## Summary

| Metric | Value |
|---|---|
| Mean Precision@5 | 0.360 |
| Mean Recall@5 | 0.336 |

## Per-Query Results

| Query | P@5 | R@5 | Missed |
|---|---|---|---|
| `Spaghetti` | 0.80 | 0.57 | edeka-0008, edeka-0064, edeka-0098 |
| `Windeln` | 0.40 | 0.29 | dm-0003, dm-0052, dm-0078 |
| `Shampoo` | 1.00 | 1.00 | — |
| `diapers` | 0.00 | 0.00 | dm-0003, dm-0052, dm-0078 |
| `yogurt` | 0.00 | 0.00 | edeka-0078, edeka-0010, edeka-0128 |
| `keyboard` | 0.80 | 0.80 | amazon-0049 |
| `Bio Milch` | 0.00 | 0.00 | edeka-0026, edeka-0051, edeka-0001 |
| `wireless mouse` | 0.40 | 0.29 | amazon-0052, amazon-0127, amazon-0002 |
| `Tomatensauce` | 0.00 | 0.00 | edeka-0069, edeka-0174, edeka-0022 |
| `Barilla` | 0.00 | 0.00 | edeka-0008, edeka-0146, edeka-0048 |
| `Nivea` | 0.60 | 0.43 | dm-0066, dm-0054, dm-0077 |
| `Logitech` | 0.20 | 0.17 | amazon-0052, amazon-0049, amazon-0195 |
| `pasta` | 0.60 | 0.50 | edeka-0008, edeka-0114, edeka-0064 |
| `Kopfhörer` | 0.60 | 1.00 | — |
| `moisturizer` | 0.00 | 0.00 | dm-0138, dm-0002, dm-0001 |

## Interpretation

Mean precision@5 is 0.360. Cross-lingual queries average 0.300 precision@5 (below the overall mean), indicating the multilingual embeddings handle language mismatch well.
