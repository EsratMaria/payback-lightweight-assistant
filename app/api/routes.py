from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.router import get_default_router
from app.config import settings
from app.models.schemas import (
    AssistRequest,
    AssistantResponse,
    Partner,
    UserContext,
)
from app.services.user_profile_store import get_default_store

logger = logging.getLogger(__name__)

api_router = APIRouter()

# ---------------------------------------------------------------------------
# Module-level singletons — initialised once at import time
# ---------------------------------------------------------------------------

_router = get_default_router()
_user_store = get_default_store()

# ---------------------------------------------------------------------------
# Catalog metadata helpers (read once at import, no LLM involved)
# ---------------------------------------------------------------------------

_CATALOGS_DIR = Path(__file__).parent.parent.parent / "data" / "catalogs"
_PARTNER_DESCRIPTIONS = {
    Partner.dm: "Drugstore: cosmetics, baby products, household, personal care",
    Partner.edeka: "Grocery: fresh food, dairy, pantry, beverages",
    Partner.amazon: "Long-tail: electronics, home, books, tools, and more",
}


@lru_cache(maxsize=1)
def _load_catalog_counts() -> dict[Partner, int]:
    counts: dict[Partner, int] = {}
    for partner in Partner:
        catalog_path = _CATALOGS_DIR / f"{partner.value}.json"
        if catalog_path.exists():
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
            counts[partner] = len(data)
        else:
            counts[partner] = 0
    return counts


# ---------------------------------------------------------------------------
# Response models for supporting endpoints
# ---------------------------------------------------------------------------


class UserSummary(BaseModel):
    user_id: str
    is_new_user: bool
    partner_affinity: dict[Partner, float]
    description: str


class PartnerInfo(BaseModel):
    partner: Partner
    name: str
    description: str
    product_count: int


class Stats(BaseModel):
    total_products: int
    products_per_partner: dict[Partner, int]
    embedding_model: str
    vector_store_backend: str
    default_llm: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PARTNER_DISPLAY_NAMES = {
    Partner.dm: "dm",
    Partner.edeka: "EDEKA",
    Partner.amazon: "Amazon",
}


def _describe_user(ctx: UserContext) -> str:
    if ctx.is_new_user:
        return "New user — no shopping history, cold-start ranking"
    dominant = max(ctx.partner_affinity, key=lambda p: ctx.partner_affinity[p])
    top_share = ctx.partner_affinity[dominant]
    if top_share >= 0.5:
        return f"{_PARTNER_DISPLAY_NAMES[dominant]}-heavy shopper ({top_share:.0%} affinity)"
    return "Balanced shopper — roughly equal affinity across partners"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@api_router.post(
    "/assist",
    response_model=AssistantResponse,
    summary="Get product recommendations or a clarifying question",
    description=(
        "Classifies the user's intent, optionally expands basket-style queries, "
        "retrieves products from the appropriate partner catalogs, applies "
        "loyalty-aware ranking, and returns either a list of recommendations, "
        "a clarifying question, or a navigation target.\n\n"
        "Pass an optional `user_id` to apply personalized partner-diversity ranking. "
        "Without `user_id`, the request is treated as a cold start (rank by query "
        "context only).\n\n"
        "Available mock user_ids for the demo: see `GET /users`."
    ),
    tags=["assistant"],
)
async def assist(request: AssistRequest) -> AssistantResponse:
    user_context: Optional[UserContext] = None
    if request.user_id:
        user_context = _user_store.get(request.user_id)
        if user_context is None:
            logger.warning(
                "Unknown user_id %r — falling back to cold start.", request.user_id
            )
    return await _router.handle(query=request.query, user_context=user_context)


@api_router.get(
    "/users",
    response_model=list[UserSummary],
    summary="List available mock user profiles",
    description=(
        "Returns all user profiles available in the demo user store. "
        "Use any returned `user_id` in a `POST /assist` request to see "
        "personalised loyalty-aware recommendations."
    ),
    tags=["users"],
)
async def list_users() -> list[UserSummary]:
    return [
        UserSummary(
            user_id=uid,
            is_new_user=ctx.is_new_user,
            partner_affinity=ctx.partner_affinity,
            description=_describe_user(ctx),
        )
        for uid, ctx in _user_store.list_profiles().items()
    ]


@api_router.get(
    "/users/{user_id}",
    response_model=UserSummary,
    summary="Get a single mock user profile",
    description="Returns the full profile for a known `user_id`. Returns 404 if not found.",
    tags=["users"],
)
async def get_user(user_id: str) -> UserSummary:
    ctx = _user_store.get(user_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    return UserSummary(
        user_id=user_id,
        is_new_user=ctx.is_new_user,
        partner_affinity=ctx.partner_affinity,
        description=_describe_user(ctx),
    )


@api_router.get(
    "/partners",
    response_model=list[PartnerInfo],
    summary="List supported shopping partners",
    description="Returns the three PAYBACK partners with descriptions and product counts from the loaded catalog.",
    tags=["metadata"],
)
async def list_partners() -> list[PartnerInfo]:
    counts = _load_catalog_counts()
    return [
        PartnerInfo(
            partner=partner,
            name=_PARTNER_DISPLAY_NAMES[partner],
            description=_PARTNER_DESCRIPTIONS[partner],
            product_count=counts[partner],
        )
        for partner in Partner
    ]


@api_router.get(
    "/stats",
    response_model=Stats,
    summary="Operational snapshot",
    description="Returns product counts, embedding model, vector store backend, and configured LLM.",
    tags=["metadata"],
)
async def stats() -> Stats:
    counts = _load_catalog_counts()
    total = sum(counts.values())
    backend_map = {"local": "local_chroma", "bigquery": "bigquery_vector_search"}
    return Stats(
        total_products=total,
        products_per_partner=counts,
        embedding_model=settings.embedding_model,
        vector_store_backend=backend_map.get(settings.vector_store, settings.vector_store),
        default_llm=f"{settings.default_llm_provider} / claude-sonnet-4-6",
    )


@api_router.get(
    "/health",
    summary="Health check",
    description="Returns `{\"status\": \"ok\"}` when the service is up.",
    tags=["health"],
)
async def health() -> dict:
    return {"status": "ok"}
