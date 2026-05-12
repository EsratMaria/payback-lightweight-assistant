from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router

app = FastAPI(
    title="PAYBACK Lightweight Assistant",
    description=(
        "Backend for PAYBACK's in-app shopping assistant. Routes natural-language "
        "queries (English or German) to product recommendations across dm, EDEKA, "
        "and Amazon, with loyalty-aware ranking and graceful clarification for "
        "vague queries.\n\n"
        "## Quick demo flow\n"
        "1. Check `/health` to confirm the API is live.\n"
        "2. Try `GET /users` to see available mock user profiles.\n"
        "3. Hit `POST /assist` with a query (and optional `user_id`) to see "
        "personalised recommendations.\n"
        "4. Compare cold-start vs known-user responses — the diversity signal "
        "should surface different partner rankings.\n\n"
        "## Architecture\n"
        "Intent agent (Claude tool-use) → conditional query expansion → "
        "ChromaDB semantic retrieval → loyalty-aware ranker → structured JSON. "
        "See the repo README for the full architecture diagram."
    ),
    version="0.1.0",
    openapi_tags=[
        {"name": "assistant", "description": "The main assistant endpoint."},
        {"name": "users", "description": "Mock user profiles for personalization demo."},
        {"name": "metadata", "description": "Catalog and system info."},
        {"name": "health", "description": "Operational health checks."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
