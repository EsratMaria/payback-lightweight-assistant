from __future__ import annotations

import pytest

from app.agents.router import Router, estimate_cost
from app.models.schemas import (
    ClarifyingQuestion,
    ExpandedQueries,
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
    prefers_deals: bool = False,
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
        prefers_deals=prefers_deals,
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
        self.last_promo_only = False

    async def search(self, query, top_k=10, partner_filter=None, promo_only=False):
        self.search_called = True
        self.search_call_count += 1
        self.search_queries.append(query)
        self.last_partner_filter = partner_filter
        self.last_promo_only = promo_only
        return self._results

    async def add(self, products) -> None:
        pass

    async def count(self) -> int:
        return len(self._results)


class MockQueryExpander:
    def __init__(self, sub_queries: list[str] | None = None) -> None:
        self._sub_queries = sub_queries or []
        self.called = False

    async def expand(self, query, language) -> ExpandedQueries:
        self.called = True
        return ExpandedQueries(sub_queries=self._sub_queries, reasoning="test")


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


@pytest.mark.asyncio
async def test_router_drops_low_relevance_subqueries(monkeypatch):
    """Sub-queries whose results all fall below expansion_min_relevance are dropped."""
    import app.config as cfg
    monkeypatch.setattr(cfg.settings, "expansion_min_relevance", 0.45)

    sub_queries = ["pasta dinner", "pasta", "glitter eyeshadow", "olive oil"]
    intent = _make_intent(
        specificity=Specificity.specific,
        confidence=0.9,
        extracted_query="pasta dinner",
        is_basket_query=True,
    )

    product_a = _sample_product(Partner.edeka)

    class MockVectorStorePerQuery:
        """Returns high-score results for known queries, low-score for the dropped one."""
        def __init__(self):
            self.search_queries: list[str] = []

        async def search(self, query, top_k=10, partner_filter=None, promo_only=False):
            self.search_queries.append(query)
            if query == "glitter eyeshadow":
                # All results below threshold — should be dropped
                return [(product_a, 0.2)]
            return [(product_a, 0.9)]

        async def add(self, products): pass
        async def count(self): return 1

    mock_store = MockVectorStorePerQuery()
    mock_expander = MockQueryExpander(sub_queries)
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=MockClarificationAgent(_sample_clarification()),
        vector_store=mock_store,
        ranker=MockRanker([_sample_recommendation()]),
        query_expander=mock_expander,
    )

    response = await router.handle("pasta dinner")

    assert response.response_type == "recommendations"
    assert response.debug_dropped_queries == ["glitter eyeshadow"]
    # All 4 sub-queries were searched, but only 1 dropped
    assert "glitter eyeshadow" in mock_store.search_queries


@pytest.mark.asyncio
async def test_router_preserves_subquery_metadata(monkeypatch):
    """debug_expanded_queries contains all proposed queries; debug_dropped_queries the dropped subset."""
    import app.config as cfg
    monkeypatch.setattr(cfg.settings, "expansion_min_relevance", 0.45)

    sub_queries = ["pasta dinner", "pasta", "setting spray", "olive oil"]
    intent = _make_intent(
        specificity=Specificity.specific,
        confidence=0.9,
        extracted_query="pasta dinner",
        is_basket_query=True,
    )

    product_a = _sample_product(Partner.edeka)

    class MockVectorStoreFiltered:
        async def search(self, query, top_k=10, partner_filter=None, promo_only=False):
            if query == "setting spray":
                return [(product_a, 0.1)]  # below threshold
            return [(product_a, 0.8)]      # above threshold

        async def add(self, products): pass
        async def count(self): return 1

    mock_expander = MockQueryExpander(sub_queries)
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=MockClarificationAgent(_sample_clarification()),
        vector_store=MockVectorStoreFiltered(),
        ranker=MockRanker([_sample_recommendation()]),
        query_expander=mock_expander,
    )

    response = await router.handle("pasta dinner")

    # All proposed queries are surfaced
    assert set(response.debug_expanded_queries) == set(sub_queries)
    # Only the below-threshold one is in dropped
    assert response.debug_dropped_queries == ["setting spray"]
    # Dropped is a strict subset of expanded
    assert set(response.debug_dropped_queries).issubset(set(response.debug_expanded_queries))


@pytest.mark.asyncio
async def test_router_applies_promo_filter_when_prefers_deals():
    """prefers_deals=True causes search to be called with promo_only=True."""
    intent = _make_intent(
        specificity=Specificity.specific,
        confidence=0.9,
        extracted_query="cheap wireless mouse",
        prefers_deals=True,
    )
    mock_store = MockVectorStore([(_sample_product(Partner.amazon), 0.85)])
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=MockClarificationAgent(_sample_clarification()),
        vector_store=mock_store,
        ranker=MockRanker([_sample_recommendation()]),
    )

    response = await router.handle("cheap wireless mouse")

    assert mock_store.last_promo_only is True
    assert response.promo_fallback is False
    assert response.response_type == "recommendations"


@pytest.mark.asyncio
async def test_router_falls_back_when_no_promo_matches():
    """If promo_only search returns no results, router retries without filter and sets promo_fallback=True."""
    intent = _make_intent(
        specificity=Specificity.specific,
        confidence=0.9,
        extracted_query="günstige Windeln",
        prefers_deals=True,
    )
    product = _sample_product(Partner.dm)

    class MockFallbackStore:
        def __init__(self):
            self.call_count = 0
            self.promo_only_calls: list[bool] = []

        async def search(self, query, top_k=10, partner_filter=None, promo_only=False):
            self.call_count += 1
            self.promo_only_calls.append(promo_only)
            if promo_only:
                return []  # promo filter finds nothing
            return [(product, 0.88)]

        async def add(self, products): pass
        async def count(self): return 1

    mock_store = MockFallbackStore()
    router = Router(
        intent_agent=MockIntentAgent(intent),
        clarification_agent=MockClarificationAgent(_sample_clarification()),
        vector_store=mock_store,
        ranker=MockRanker([_sample_recommendation()]),
    )

    response = await router.handle("günstige Windeln")

    assert mock_store.call_count == 2
    assert mock_store.promo_only_calls == [True, False]
    assert response.promo_fallback is True
    assert response.response_type == "recommendations"
