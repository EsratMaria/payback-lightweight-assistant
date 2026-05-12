from __future__ import annotations

import re

import pytest
from pydantic import BaseModel, ValidationError

from app.agents.query_expander import QueryExpander
from app.llm.base import LLMClient
from app.models.schemas import ExpandedQueries, Language, Partner, Product


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockLLMClient(LLMClient):
    """Returns canned ExpandedQueries instances keyed by query substring."""

    def __init__(self, responses: dict[str, ExpandedQueries]) -> None:
        self._responses = responses
        self.last_prompt: str = ""

    async def structured_completion(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> BaseModel:
        self.last_prompt = prompt
        match = re.search(r'QUERY: "(.+?)"', prompt)
        query = match.group(1) if match else prompt
        for key, result in self._responses.items():
            if key in query:
                return result
        raise AssertionError(f"MockLLMClient: no canned response for {query!r}")


class MockVectorStore:
    """Configurable mock vector store — returns a preset list of (Product, score) tuples."""

    def __init__(self, results: list[tuple[Product, float]] | None = None) -> None:
        self._results = results or []
        self.search_call_count = 0
        self.last_query: str = ""

    async def search(self, query: str, top_k: int = 10, partner_filter=None):
        self.search_call_count += 1
        self.last_query = query
        return self._results

    async def add(self, products) -> None:
        pass

    async def count(self) -> int:
        return len(self._results)


def _product(pid: str, category: str, partner: Partner = Partner.dm) -> Product:
    return Product(
        product_id=pid,
        partner=partner,
        name=f"Product {pid}",
        description="Test",
        category=category,
        price_eur=5.0,
    )


# ---------------------------------------------------------------------------
# Existing tests (updated for new constructor and ExpandedQueries return type)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basket_query_expansion():
    sub_queries = ["pasta dinner", "pasta", "tomato sauce", "olive oil", "parmesan"]
    mock_llm = MockLLMClient({
        "pasta dinner": ExpandedQueries(
            sub_queries=sub_queries,
            reasoning="Pasta dinner requires pasta, sauce, oil, and cheese.",
        )
    })
    mock_store = MockVectorStore()
    expander = QueryExpander(mock_llm, mock_store)
    result = await expander.expand("pasta dinner", Language.en)

    assert isinstance(result, ExpandedQueries)
    assert 3 <= len(result.sub_queries) <= 5
    assert result.sub_queries[0] == "pasta dinner"
    assert "pasta" in result.sub_queries


@pytest.mark.asyncio
async def test_german_basket_expansion():
    sub_queries = ["Geburtstagsparty", "Chips", "Sekt", "Kerzen"]
    mock_llm = MockLLMClient({
        "Geburtstagsparty": ExpandedQueries(
            sub_queries=sub_queries,
            reasoning="Geburtstagsparty braucht Snacks, Getränke und Deko.",
        )
    })
    mock_store = MockVectorStore()
    expander = QueryExpander(mock_llm, mock_store)
    result = await expander.expand("Geburtstagsparty", Language.de)

    assert result.sub_queries[0] == "Geburtstagsparty"
    assert len(result.sub_queries) >= 3
    assert result.reasoning != ""


def test_min_length_validation():
    with pytest.raises(ValidationError):
        ExpandedQueries(
            sub_queries=["only", "two"],
            reasoning="Too few sub-queries.",
        )


# ---------------------------------------------------------------------------
# New tests: catalog-grounding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expander_calls_preflight_retrieval():
    """_discover_available_categories is called exactly once before the LLM expansion."""
    mock_llm = MockLLMClient({
        "pasta": ExpandedQueries(
            sub_queries=["pasta dinner", "pasta", "tomato sauce"],
            reasoning="test",
        )
    })
    mock_store = MockVectorStore()  # returns [] — no categories
    expander = QueryExpander(mock_llm, mock_store)

    await expander.expand("pasta dinner", Language.en)

    # One pre-flight call, no additional store calls from expand() itself
    assert mock_store.search_call_count == 1
    assert mock_store.last_query == "pasta dinner"


@pytest.mark.asyncio
async def test_expander_passes_categories_to_prompt():
    """Categories discovered in pre-flight appear sorted and deduped in the LLM prompt."""
    products = [
        (_product("p1", "cosmetics"), 0.9),
        (_product("p2", "personal_care"), 0.8),
        (_product("p3", "cosmetics"), 0.7),   # duplicate category — should be deduped
    ]
    mock_llm = MockLLMClient({
        "glam": ExpandedQueries(
            sub_queries=["glam makeup", "cosmetics", "personal care"],
            reasoning="test",
        )
    })
    mock_store = MockVectorStore(products)
    expander = QueryExpander(mock_llm, mock_store)

    await expander.expand("glam makeup", Language.en)

    # Sorted + deduped: ["cosmetics", "personal_care"]
    assert "cosmetics, personal_care" in mock_llm.last_prompt


@pytest.mark.asyncio
async def test_expander_handles_empty_preflight():
    """When pre-flight returns nothing, the prompt contains the fallback hint."""
    mock_llm = MockLLMClient({
        "glam": ExpandedQueries(
            sub_queries=["glam makeup", "foundation", "lipstick"],
            reasoning="test",
        )
    })
    mock_store = MockVectorStore([])  # empty catalog response
    expander = QueryExpander(mock_llm, mock_store)

    await expander.expand("glam makeup", Language.en)

    assert (
        "no clear category matches found in the catalog for this query"
        in mock_llm.last_prompt
    )
