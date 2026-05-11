from __future__ import annotations

import pytest

from app.agents.router import Router, estimate_cost
from app.models.schemas import (
    ClarifyingQuestion,
    Intent,
    IntentResult,
    Language,
    Partner,
    Product,
    ProductRecommendation,
    Specificity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(
    intent: Intent = Intent.search,
    specificity: Specificity = Specificity.specific,
    language: Language = Language.en,
    confidence: float = 0.9,
    target_partner: Partner | None = None,
    extracted_query: str = "test query",
    reasoning: str = "test",
    is_basket_query: bool = False,
) -> IntentResult:
    return IntentResult(
        language=language,
        intent=intent,
        specificity=specificity,
        confidence=confidence,
        extracted_query=extracted_query,
        target_partner=target_partner,
        reasoning=reasoning,
        is_basket_query=is_basket_query,
    )


def _sample_product(partner: Partner = Partner.edeka) -> Product:
    return Product(
        product_id="test-001",
        partner=partner,
        name="Test Product",
        description="A test product",
        category="grocery",
        price_eur=2.99,
    )


def _sample_recommendation(rank: int = 1) -> ProductRecommendation:
    return ProductRecommendation(
        product=_sample_product(),
        semantic_score=0.8,
        loyalty_boost=0.0,
        final_score=0.8,
        rank=rank,
    )


def _sample_clarification() -> ClarifyingQuestion:
    return ClarifyingQuestion(
        question="What kind of product are you looking for?",
        suggested_options=["Food", "Household", "Personal care"],
    )


# ---------------------------------------------------------------------------
# Mock dependencies
# ---------------------------------------------------------------------------


class MockIntentAgent:
    def __init__(self, result: IntentResult) -> None:
        self._result = result

    async def classify(self, query: str) -> IntentResult:
        return self._result


class MockClarificationAgent:
    def __init__(self, result: ClarifyingQuestion) -> None:
        self._result = result
        self.called = False

    async def generate(self, query, top_results, language) -> ClarifyingQuestion:
        self.called = True
        return self._result


class MockVectorStore:
    def __init__(self, results: list[tuple[Product, float]] | None = None) -> None:
        self._results = results or []
        self.search_called = False
        self.search_call_count = 0
        self.search_queries: list[str] = []
        self.last_partner_filter = "not_called"

    async def search(self, query, top_k=10, partner_filter=None):
        self.search_called = True
        self.search_call_count += 1
        self.search_queries.append(query)
        self.last_partner_filter = partner_filter
        return self._results

    async def add(self, products) -> None:
        pass

    async def count(self) -> int:
        return len(self._results)


class MockQueryExpander:
    def __init__(self, sub_queries: list[str] | None = None) -> None:
        self._sub_queries = sub_queries or []
        self.called = False

    async def expand(self, query, language) -> list[str]:
        self.called = True
        return self._sub_queries


class MockRanker:
    def __init__(self, recommendations: list[ProductRecommendation] | None = None) -> None:
        self._recommendations = recommendations or [_sample_recommendation()]

    async def rank(self, products_with_scores, user_context):
        return self._recommendations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_navigational_path():
    intent = _make_intent(
        specificity=Specificity.navigational,
        target_partner=Partner.amazon,
    )
    mock_store = MockVectorStore()
    mock_clarification = MockClarificationAgent(_sample_clarification())
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=mock_clarification,
        vector_store=mock_store,
        ranker=MockRanker(),
    )

    response = await router.handle("open Amazon for me")

    assert response.response_type == "navigation"
    assert response.navigation_target == "amazon"
    assert mock_store.search_called is False
    assert mock_clarification.called is False


@pytest.mark.asyncio
async def test_specific_high_confidence_path():
    intent = _make_intent(specificity=Specificity.specific, confidence=0.9)
    results = [(_sample_product(), 0.82)]
    mock_store = MockVectorStore(results)
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=MockClarificationAgent(_sample_clarification()),
        vector_store=mock_store,
        ranker=MockRanker([_sample_recommendation()]),
    )

    response = await router.handle("pasta dinner")

    assert response.response_type == "recommendations"
    assert response.recommendations is not None
    assert len(response.recommendations) >= 1


@pytest.mark.asyncio
async def test_specific_low_confidence_falls_to_clarification():
    intent = _make_intent(specificity=Specificity.specific, confidence=0.3)
    mock_clarification = MockClarificationAgent(_sample_clarification())
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=mock_clarification,
        vector_store=MockVectorStore(),
        ranker=MockRanker(),
    )

    response = await router.handle("pasta dinner")

    assert response.response_type == "clarification"
    assert response.clarification is not None
    assert mock_clarification.called is True


@pytest.mark.asyncio
async def test_vague_path():
    intent = _make_intent(specificity=Specificity.vague, confidence=0.85)
    mock_store = MockVectorStore()
    mock_clarification = MockClarificationAgent(_sample_clarification())
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=mock_clarification,
        vector_store=mock_store,
        ranker=MockRanker(),
    )

    response = await router.handle("something for my dog")

    assert response.response_type == "clarification"
    assert mock_store.search_called is True
    assert mock_clarification.called is True


@pytest.mark.asyncio
async def test_partner_filter_applied_when_target_partner_set():
    intent = _make_intent(
        specificity=Specificity.specific,
        confidence=0.9,
        target_partner=Partner.edeka,
    )
    mock_store = MockVectorStore([(_sample_product(Partner.edeka), 0.85)])
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=MockClarificationAgent(_sample_clarification()),
        vector_store=mock_store,
        ranker=MockRanker(),
    )

    await router.handle("pasta dinner from edeka")

    assert mock_store.last_partner_filter == Partner.edeka


@pytest.mark.asyncio
async def test_latency_and_cost_populated():
    intent = _make_intent(specificity=Specificity.specific, confidence=0.9)
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=MockClarificationAgent(_sample_clarification()),
        vector_store=MockVectorStore([(_sample_product(), 0.8)]),
        ranker=MockRanker(),
    )

    response = await router.handle("wireless mouse")

    assert response.latency_ms > 0
    assert response.estimated_cost_eur > 0


@pytest.mark.asyncio
async def test_support_intent_returns_out_of_scope_message():
    intent = _make_intent(
        intent=Intent.support,
        specificity=Specificity.specific,
        confidence=0.97,
        language=Language.en,
    )
    mock_store = MockVectorStore()
    mock_clarification = MockClarificationAgent(_sample_clarification())
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=mock_clarification,
        vector_store=mock_store,
        ranker=MockRanker(),
    )

    response = await router.handle("how do I redeem my points")

    assert response.response_type == "clarification"
    assert "payback" in response.clarification.question.lower()
    assert mock_store.search_called is False
    assert mock_clarification.called is False


@pytest.mark.asyncio
async def test_basket_query_triggers_expansion():
    """is_basket_query=True causes the expander to run and search to be called once per sub-query."""
    sub_queries = ["pasta dinner", "pasta", "tomato sauce", "olive oil"]
    intent = _make_intent(
        specificity=Specificity.specific,
        confidence=0.9,
        extracted_query="pasta dinner",
        is_basket_query=True,
    )
    product_a = _sample_product(Partner.edeka)
    product_b = Product(
        product_id="test-002", partner=Partner.edeka,
        name="Tomato Sauce", description="Sauce", category="pantry", price_eur=1.5,
    )
    mock_store = MockVectorStore()
    mock_store._results = [(product_a, 0.8)]
    mock_expander = MockQueryExpander(sub_queries)
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=MockClarificationAgent(_sample_clarification()),
        vector_store=mock_store,
        ranker=MockRanker([_sample_recommendation()]),
        query_expander=mock_expander,
    )

    response = await router.handle("pasta dinner")

    assert mock_expander.called is True
    assert mock_store.search_call_count == len(sub_queries)
    assert set(mock_store.search_queries) == set(sub_queries)
    assert response.debug_expanded_queries == sub_queries


@pytest.mark.asyncio
async def test_single_item_query_skips_expansion():
    """is_basket_query=False: expander is never called, search called exactly once."""
    intent = _make_intent(
        specificity=Specificity.specific,
        confidence=0.9,
        extracted_query="wireless mouse",
        is_basket_query=False,
    )
    mock_store = MockVectorStore([(_sample_product(), 0.8)])
    mock_expander = MockQueryExpander(["this", "should", "not", "be", "called"])
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=MockClarificationAgent(_sample_clarification()),
        vector_store=mock_store,
        ranker=MockRanker([_sample_recommendation()]),
        query_expander=mock_expander,
    )

    await router.handle("wireless mouse")

    assert mock_expander.called is False
    assert mock_store.search_call_count == 1


@pytest.mark.asyncio
async def test_navigational_without_partner_asks_which_partner():
    intent = _make_intent(
        specificity=Specificity.navigational,
        target_partner=None,
        language=Language.en,
    )
    mock_store = MockVectorStore()
    mock_clarification = MockClarificationAgent(_sample_clarification())
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=mock_clarification,
        vector_store=mock_store,
        ranker=MockRanker(),
    )

    response = await router.handle("take me to the shop")

    assert response.response_type == "clarification"
    assert response.clarification.suggested_options == ["dm", "EDEKA", "Amazon"]
    assert mock_store.search_called is False
    assert mock_clarification.called is False
