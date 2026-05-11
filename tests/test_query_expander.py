from __future__ import annotations

import re

import pytest
from pydantic import BaseModel, ValidationError

from app.agents.query_expander import QueryExpander
from app.llm.base import LLMClient
from app.models.schemas import ExpandedQueries, Language


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------


class MockLLMClient(LLMClient):
    """Returns canned ExpandedQueries instances keyed by query substring."""

    def __init__(self, responses: dict[str, ExpandedQueries]) -> None:
        self._responses = responses

    async def structured_completion(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> BaseModel:
        match = re.search(r'QUERY: "(.+?)"', prompt)
        query = match.group(1) if match else prompt
        for key, result in self._responses.items():
            if key in query:
                return result
        raise AssertionError(f"MockLLMClient: no canned response for {query!r}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basket_query_expansion():
    sub_queries = ["pasta dinner", "pasta", "tomato sauce", "olive oil", "parmesan"]
    mock = MockLLMClient({
        "pasta dinner": ExpandedQueries(
            sub_queries=sub_queries,
            reasoning="Pasta dinner requires pasta, sauce, oil, and cheese.",
        )
    })
    expander = QueryExpander(mock)
    result = await expander.expand("pasta dinner", Language.en)

    assert 3 <= len(result) <= 5
    assert result[0] == "pasta dinner"
    assert "pasta" in result


@pytest.mark.asyncio
async def test_german_basket_expansion():
    sub_queries = ["Geburtstagsparty", "Chips", "Sekt", "Kerzen"]
    mock = MockLLMClient({
        "Geburtstagsparty": ExpandedQueries(
            sub_queries=sub_queries,
            reasoning="Geburtstagsparty braucht Snacks, Getränke und Deko.",
        )
    })
    expander = QueryExpander(mock)
    result = await expander.expand("Geburtstagsparty", Language.de)

    assert result[0] == "Geburtstagsparty"
    assert len(result) >= 3
    # Verify the canned reasoning is accessible through the mock (sanity check)
    raw = await mock.structured_completion(
        prompt='QUERY: "Geburtstagsparty"', schema=ExpandedQueries
    )
    assert raw.reasoning != ""


def test_min_length_validation():
    with pytest.raises(ValidationError):
        ExpandedQueries(
            sub_queries=["only", "two"],
            reasoning="Too few sub-queries.",
        )
