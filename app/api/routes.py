from __future__ import annotations

from fastapi import APIRouter

from app.agents.router import get_default_router
from app.models.schemas import AssistRequest, AssistantResponse

router = APIRouter()

_router = get_default_router()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/assist", response_model=AssistantResponse)
async def assist(request: AssistRequest) -> AssistantResponse:
    return await _router.handle(query=request.query, user_context=request.user_context)
