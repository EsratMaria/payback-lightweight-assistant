# Retrieval Eval (Precision@5 / Recall@5) — 20260514_145426

**Dataset:** 15 queries  
**Duration:** 10.7s  

## Summary

| Metric | Value |
|---|---|
| Mean Precision@5 | 0.360 |
| Mean Recall@5 | 0.336 |

## Per-Query Results

| Query | P@5 | R@5 | Missed |
|---|---|---|---|
| `Spaghetti` | 0.80 | 0.57 | edeka-0008, edeka-0098, edeka-0064 |
| `Windeln` | 0.40 | 0.29 | dm-0027, dm-0039, dm-0052 |
| `Shampoo` | 1.00 | 1.00 | — |
| `diapers` | 0.00 | 0.00 | dm-0027, dm-0014, dm-0039 |
| `yogurt` | 0.00 | 0.00 | edeka-0017, edeka-0078, edeka-0128 |
| `keyboard` | 0.80 | 0.80 | amazon-0049 |
| `Bio Milch` | 0.00 | 0.00 | edeka-0076, edeka-0101, edeka-0001 |
| `wireless mouse` | 0.40 | 0.29 | amazon-0052, amazon-0102, amazon-0127 |
| `Tomatensauce` | 0.00 | 0.00 | edeka-0022, edeka-0069, edeka-0149 |
| `Barilla` | 0.00 | 0.00 | edeka-0008, edeka-0048, edeka-0098 |
| `Nivea` | 0.60 | 0.43 | dm-0054, dm-0032, dm-0066 |
| `Logitech` | 0.20 | 0.17 | amazon-0027, amazon-0052, amazon-0195 |
| `pasta` | 0.60 | 0.50 | edeka-0008, edeka-0114, edeka-0064 |
| `Kopfhörer` | 0.60 | 1.00 | — |
| `moisturizer` | 0.00 | 0.00 | dm-0002, dm-0032, dm-0051 |

## Interpretation

Mean precision@5 is 0.360. Cross-lingual queries average 0.300 precision@5 (below the overall mean), indicating the multilingual embeddings handle language mismatch well.
