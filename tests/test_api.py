"""API endpoint tests.

Uses FastAPI's TestClient with the real app. LLM calls are intercepted via
monkeypatching app.api.routes._router with a MockRouter so tests are fast,
free, and deterministic.

Tests 1-3 patch the router; tests 4-11 don't touch it (validation / GET only).
"""
from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes_module
from app.models.schemas import (
    AssistantResponse,
    ClarifyingQuestion,
    Intent,
    IntentResult,
    Language,
    Partner,
    ProductRecommendation,
    Specificity,
    Product,
    UserContext,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_intent(language: Language = Language.en) -> IntentResult:
    return IntentResult(
        language=language,
        intent=Intent.search,
        specificity=Specificity.specific,
        confidence=0.9,
        extracted_query="test query",
        reasoning="test",
    )


def _make_product(partner: Partner, pid: str = "test-001") -> Product:
    return Product(
        product_id=pid,
        partner=partner,
        name=f"Test Product {pid}",
        description="A test product",
        category="grocery",
        price_eur=2.99,
    )


def _make_recommendation(partner: Partner, rank: int = 1) -> ProductRecommendation:
    return ProductRecommendation(
        product=_make_product(partner, f"pid-{rank}"),
        semantic_score=0.8,
        loyalty_boost=0.0,
        diversity_bonus=0.0,
        final_score=0.8,
        rank=rank,
    )


def _recommendations_response(
    partner: Partner = Partner.edeka,
    language: Language = Language.en,
) -> AssistantResponse:
    return AssistantResponse(
        response_type="recommendations",
        intent_result=_make_intent(language),
        recommendations=[_make_recommendation(partner, i) for i in range(1, 4)],
        latency_ms=100.0,
        estimated_cost_eur=0.001,
    )


class MockRouter:
    """Records the most-recent call; returns configurable response."""

    def __init__(self, response: AssistantResponse) -> None:
        self._response = response
        self.last_user_context: Optional[UserContext] = "not_called"  # sentinel

    async def handle(
        self, query: str, user_context: Optional[UserContext] = None
    ) -> AssistantResponse:
        self.last_user_context = user_context
        return self._response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(routes_module.app if hasattr(routes_module, "app") else __import__("app.main", fromlist=["app"]).app)


@pytest.fixture(autouse=False)
def mock_router_cold_start(monkeypatch):
    """Patch routes._router with a mock returning an EDEKA recommendation."""
    mock = MockRouter(_recommendations_response(Partner.edeka))
    monkeypatch.setattr(routes_module, "_router", mock)
    return mock


@pytest.fixture(autouse=False)
def mock_router_amazon(monkeypatch):
    """Patch routes._router with a mock returning an Amazon recommendation."""
    mock = MockRouter(_recommendations_response(Partner.amazon))
    monkeypatch.setattr(routes_module, "_router", mock)
    return mock


# ---------------------------------------------------------------------------
# Import the FastAPI app after module setup
# ---------------------------------------------------------------------------

from app.main import app  # noqa: E402

_test_client = TestClient(app)


# ---------------------------------------------------------------------------
# Test 1: POST /assist — cold start (no user_id)
# ---------------------------------------------------------------------------


def test_assist_with_query_only(monkeypatch):
    mock = MockRouter(_recommendations_response(Partner.edeka))
    monkeypatch.setattr(routes_module, "_router", mock)

    resp = _test_client.post("/assist", json={"query": "pasta dinner"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["response_type"] == "recommendations"
    assert body["recommendations"] is not None
    assert len(body["recommendations"]) > 0
    # Router was called with no user_context (cold start)
    assert mock.last_user_context is None


# ---------------------------------------------------------------------------
# Test 2: POST /assist — known user_id resolves to UserContext
# ---------------------------------------------------------------------------


def test_assist_with_known_user_id(monkeypatch):
    # Return different recommendations depending on whether user_context is set
    cold_resp = _recommendations_response(Partner.edeka)
    personalized_resp = _recommendations_response(Partner.amazon)

    class ContextAwareMock:
        def __init__(self):
            self.last_user_context = None

        async def handle(self, query, user_context=None):
            self.last_user_context = user_context
            return personalized_resp if user_context else cold_resp

    mock = ContextAwareMock()
    monkeypatch.setattr(routes_module, "_router", mock)

    resp = _test_client.post(
        "/assist", json={"query": "pasta dinner", "user_id": "user_edeka_heavy"}
    )

    assert resp.status_code == 200
    # The router was called with a real UserContext (not None)
    assert mock.last_user_context is not None
    assert isinstance(mock.last_user_context, UserContext)
    assert mock.last_user_context.is_new_user is False
    # Partner affinity matches user_edeka_heavy profile
    assert mock.last_user_context.partner_affinity[Partner.edeka] == pytest.approx(0.7)
    # Response reflects personalized path
    body = resp.json()
    assert body["response_type"] == "recommendations"
    assert body["recommendations"][0]["product"]["partner"] == "amazon"


# ---------------------------------------------------------------------------
# Test 3: POST /assist — unknown user_id → 200 + cold start fallback
# ---------------------------------------------------------------------------


def test_assist_with_unknown_user_id(monkeypatch):
    mock = MockRouter(_recommendations_response(Partner.edeka))
    monkeypatch.setattr(routes_module, "_router", mock)

    resp = _test_client.post(
        "/assist", json={"query": "pasta dinner", "user_id": "bogus_user_xyz"}
    )

    assert resp.status_code == 200  # NOT 404 — soft fallback
    body = resp.json()
    assert body["response_type"] == "recommendations"
    # Router was called with None (cold start fallback)
    assert mock.last_user_context is None


# ---------------------------------------------------------------------------
# Test 4: Validation — empty query → 422
# ---------------------------------------------------------------------------


def test_assist_validation_empty_query():
    resp = _test_client.post("/assist", json={"query": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 5: Validation — query too long → 422
# ---------------------------------------------------------------------------


def test_assist_validation_long_query():
    resp = _test_client.post("/assist", json={"query": "x" * 600})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test 6: GET /users — lists all profiles
# ---------------------------------------------------------------------------


def test_users_endpoint_lists_profiles():
    resp = _test_client.get("/users")
    assert resp.status_code == 200
    users = resp.json()
    assert len(users) >= 3
    user_ids = {u["user_id"] for u in users}
    assert "user_balanced" in user_ids
    assert "user_edeka_heavy" in user_ids
    assert "user_new" in user_ids
    # Each user has required fields
    for u in users:
        assert "user_id" in u
        assert "is_new_user" in u
        assert "partner_affinity" in u
        assert "description" in u
        assert len(u["description"]) > 0


# ---------------------------------------------------------------------------
# Test 7: GET /users/{user_id} — returns correct profile
# ---------------------------------------------------------------------------


def test_user_detail_endpoint():
    resp = _test_client.get("/users/user_balanced")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "user_balanced"
    assert body["is_new_user"] is False
    assert body["partner_affinity"]["edeka"] == pytest.approx(0.33)
    assert body["partner_affinity"]["dm"] == pytest.approx(0.33)
    assert body["partner_affinity"]["amazon"] == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# Test 8: GET /users/{user_id} — unknown → 404
# ---------------------------------------------------------------------------


def test_user_detail_unknown_404():
    resp = _test_client.get("/users/nonexistent_user")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test 9: GET /partners — all three with product_count
# ---------------------------------------------------------------------------


def test_partners_endpoint():
    resp = _test_client.get("/partners")
    assert resp.status_code == 200
    partners = resp.json()
    assert len(partners) == 3
    partner_names = {p["partner"] for p in partners}
    assert partner_names == {"dm", "edeka", "amazon"}
    for p in partners:
        assert p["product_count"] > 0
        assert len(p["description"]) > 0
        assert len(p["name"]) > 0


# ---------------------------------------------------------------------------
# Test 10: GET /stats — totals are consistent
# ---------------------------------------------------------------------------


def test_stats_endpoint():
    resp = _test_client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_products"] > 0
    per_partner_total = sum(body["products_per_partner"].values())
    assert per_partner_total == body["total_products"]
    assert len(body["embedding_model"]) > 0
    assert len(body["vector_store_backend"]) > 0
    assert len(body["default_llm"]) > 0


# ---------------------------------------------------------------------------
# Test 11: GET /health — still works
# ---------------------------------------------------------------------------


def test_health_still_works():
    resp = _test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
