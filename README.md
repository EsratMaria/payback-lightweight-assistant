# PAYBACK Lightweight Assistant

## Overview

Backend microservice that accepts natural-language queries in German or English, classifies
intent via an LLM (Claude or Gemini), retrieves relevant products from a local ChromaDB
vector store, re-ranks them with a loyalty-aware scoring function, and returns structured
JSON — powering the in-app product discovery feature for the PAYBACK loyalty programme.

## Architecture

<!-- TODO: add architecture diagram to docs/ -->

See [ARCHITECTURE.md](ARCHITECTURE.md) and [docs/decisions.md](docs/decisions.md).

## Quick Start

```bash
# 1. Install dependencies
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure keys
cp .env.example .env   # then edit .env with your ANTHROPIC_API_KEY / GOOGLE_API_KEY

# 3. Generate product catalogs (~700 products via Claude)
python data/generate_catalogs.py

# 4. Run the API
uvicorn app.main:app --reload
# → http://localhost:8000/docs

# 5. Run tests
pytest tests/ -v

# 6. Run the demo notebook
jupyter notebook notebooks/demo.ipynb
```

## Design Decisions

See [docs/decisions.md](docs/decisions.md) for all Architecture Decision Records.

## Cloud Deployment

<!-- TODO: Cloud Run deployment instructions -->

Run `./scripts/deploy_gcp.sh <PROJECT_ID>` after setting up GCP credentials.

## Cost & Performance

<!-- TODO: benchmark results and cost breakdown -->

See `scripts/cost_analysis.py` for per-provider cost estimates and `scripts/load_test.py`
for latency benchmarks.
