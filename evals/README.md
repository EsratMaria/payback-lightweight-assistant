# PAYBACK Assistant — Evaluation Suite

Three reproducible metrics for the shopping assistant:

| Eval | Metric | Dataset | Runner |
|---|---|---|---|
| Intent classification | Per-dimension accuracy | `datasets/intent_labels.yaml` (25 queries) | `run_intent_eval.py` |
| Retrieval quality | Precision@5 / Recall@5 | `datasets/retrieval_labels.yaml` (15 queries) | `run_retrieval_eval.py` |
| End-to-end quality | LLM-as-judge (0-3 per dim) | `datasets/judge_validation.yaml` (10 queries) | `run_e2e_eval.py` |

---

## Running the evals

Prerequisite: catalogs must be ingested before retrieval and E2E evals will work.

```bash
python -m app.retrieval.ingest --rebuild
```

Then run from the repo root:

```bash
# Intent accuracy (calls real Claude Sonnet — ~25 API calls)
python evals/run_intent_eval.py

# Retrieval precision@5 / recall@5 (no LLM calls — uses ChromaDB only)
python evals/run_retrieval_eval.py

# E2E judge eval (calls Claude Opus — 3 runs × 10 queries = 30 Opus API calls)
python evals/run_e2e_eval.py
```

Each run writes a timestamped report to `evals/results/<eval>_<timestamp>/`:
- `report.md` — human-readable markdown
- `result.json` — machine-readable full result

---

## Judge model rationale

The production system uses **Claude Sonnet** (`claude-sonnet-4-6`).
The judge uses **Claude Opus** (`claude-opus-4-7`).

Deliberately different models reduce self-preference bias: a model that generated
a response tends to rate its own outputs more favourably. Using a larger, different
model as the judge is the standard mitigation within a single-provider setup.
The ideal mitigation is cross-provider evaluation (e.g. Gemini or GPT-4 as judge),
which is on the roadmap.

---

## Judge–human agreement validation

To validate that the Opus judge aligns with human judgment:

1. Run `python evals/run_e2e_eval.py` once to get system outputs for the 10 validation queries.
2. Open `evals/datasets/judge_validation.yaml` and fill in `human_rating_*` fields (0-3 scale):
   - `0` = completely off / frustrating
   - `1` = partial / has issues
   - `2` = mostly good
   - `3` = excellent, would ship
3. Re-run `python evals/run_e2e_eval.py`.
4. The report will populate the **Judge–Human Agreement** section.
   - Target: ≥75% within-1-point agreement per dimension.
   - Below 70% on any dimension: the judge prompt needs revision.

---

## Known limitations

1. **Single-provider judge.** Opus and Sonnet are both Anthropic models — correlated biases exist.
2. **Small datasets.** 25/15/10 queries are sufficient for early signal, not statistical confidence.
3. **Static ground truth.** `retrieval_labels.yaml` product IDs will drift as catalogs change.
4. **No user feedback loop.** Judge scores are a proxy for user satisfaction, not a substitute.
5. **Intent labels are auto-generated.** Review `datasets/intent_labels.yaml` before treating accuracy numbers as authoritative.

---

## TODO for Maria

- [ ] Review auto-generated labels in `datasets/intent_labels.yaml` — verify `is_basket_query` and `specificity` for edge cases.
- [ ] Check `datasets/retrieval_labels.yaml` — confirm product IDs are still in the catalog after any re-generation.
- [ ] Run the E2E eval once, then hand-rate the 10 queries in `datasets/judge_validation.yaml`.
- [ ] Re-run `python evals/run_e2e_eval.py` to get the judge–human agreement number.
