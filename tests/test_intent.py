from __future__ import annotations

import os
import re

import pytest
from pydantic import BaseModel

from app.agents.intent_agent import IntentAgent, get_default_intent_agent
from app.llm.base import LLMClient
from app.llm.claude_client import ClaudeClient
from app.models.schemas import Intent, IntentResult, Language, Partner, Specificity


# ---------------------------------------------------------------------------
# Mock LLM client — no real API calls
# ---------------------------------------------------------------------------


class MockLLMClient(LLMClient):
    """Deterministic LLM client for unit tests.

    Takes a dict mapping query substrings to canned IntentResult instances.
    Extracts the query from the prompt (looks for QUERY: "..." pattern),
    finds a matching key, and returns the canned result.
    Raises AssertionError on unexpected queries so tests fail loudly.
    """

    def __init__(self, responses: dict[str, IntentResult]) -> None:
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

        raise AssertionError(
            f"MockLLMClient: no canned response for query {query!r}. "
            f"Registered keys: {list(self._responses.keys())}"
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
    extracted_query: str = "query",
    reasoning: str = "test reasoning",
) -> IntentResult:
    return IntentResult(
        language=language,
        intent=intent,
        specificity=specificity,
        confidence=confidence,
        extracted_query=extracted_query,
        target_partner=target_partner,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_specific_english_query():
    mock = MockLLMClient({
        "pasta dinner": _make_intent(
            intent=Intent.search,
            specificity=Specificity.specific,
            language=Language.en,
            confidence=0.9,
            extracted_query="pasta dinner",
            reasoning="Specific product search in English.",
        )
    })
    agent = IntentAgent(mock)
    import asyncio
    result = asyncio.run(agent.classify("I need stuff for a pasta dinner"))

    assert result.intent == Intent.search
    assert result.specificity == Specificity.specific
    assert result.language == Language.en
    assert result.confidence == 0.9


def test_vague_query():
    mock = MockLLMClient({
        "something for my dog": _make_intent(
            specificity=Specificity.vague,
            confidence=0.85,
            reasoning="Too broad — no product category or partner implied.",
        )
    })
    agent = IntentAgent(mock)
    import asyncio
    result = asyncio.run(agent.classify("something for my dog"))

    assert result.specificity == Specificity.vague
    assert result.reasoning != ""


def test_navigational_query():
    mock = MockLLMClient({
        "open Amazon": _make_intent(
            specificity=Specificity.navigational,
            target_partner=Partner.amazon,
            extracted_query="",
            reasoning="User explicitly names Amazon — navigational.",
        )
    })
    agent = IntentAgent(mock)
    import asyncio
    result = asyncio.run(agent.classify("open Amazon for me"))

    assert result.specificity == Specificity.navigational
    assert result.target_partner == Partner.amazon


def test_german_query():
    mock = MockLLMClient({
        "Windeln": _make_intent(
            language=Language.de,
            specificity=Specificity.specific,
            extracted_query="günstige Windeln",
            reasoning="German-language specific product query.",
        )
    })
    agent = IntentAgent(mock)
    import asyncio
    result = asyncio.run(
        agent.classify("Bitte zeige mir Angebote für günstige Windeln")
    )

    assert result.language == Language.de
    assert result.specificity == Specificity.specific


def test_partner_mention_in_specific_query():
    """When the user mentions a recognized partner in a specific query, target_partner is set."""
    mock = MockLLMClient({
        "pasta dinner from edeka": _make_intent(
            intent=Intent.search,
            specificity=Specificity.specific,
            language=Language.en,
            confidence=0.92,
            target_partner=Partner.edeka,
            extracted_query="pasta dinner",
            reasoning="Specific search query with explicit edeka partner mention.",
        )
    })
    agent = IntentAgent(mock)
    import asyncio
    result = asyncio.run(agent.classify("pasta dinner from edeka"))

    assert result.intent == Intent.search
    assert result.specificity == Specificity.specific
    assert result.target_partner == Partner.edeka


def test_unrecognized_partner_falls_through_to_null():
    """When the user mentions a partner not in our system, target_partner stays null."""
    mock = MockLLMClient({
        "pasta from REWE": _make_intent(
            intent=Intent.search,
            specificity=Specificity.specific,
            language=Language.en,
            confidence=0.88,
            target_partner=None,
            extracted_query="pasta",
            reasoning="REWE is not a supported partner; falling through to search across available partners.",
        )
    })
    agent = IntentAgent(mock)
    import asyncio
    result = asyncio.run(agent.classify("pasta from REWE"))

    assert result.target_partner is None
    assert result.reasoning != ""


def test_factory_claude(monkeypatch):
    import app.config
    monkeypatch.setattr(app.config.settings, "default_llm_provider", "claude")
    agent = get_default_intent_agent()
    assert isinstance(agent.llm, ClaudeClient)


def test_factory_gemini_raises(monkeypatch):
    import app.config
    monkeypatch.setattr(app.config.settings, "default_llm_provider", "gemini")
    with pytest.raises(NotImplementedError):
        get_default_intent_agent()


def test_factory_unknown_raises(monkeypatch):
    import app.config
    monkeypatch.setattr(app.config.settings, "default_llm_provider", "magic-llm-9000")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_default_intent_agent()


# ---------------------------------------------------------------------------
# Integration test — skipped unless UNIFIED_ENDPOINT_KEY is set
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("UNIFIED_ENDPOINT_KEY"),
    reason="needs UNIFIED_ENDPOINT_KEY in env",
)
def test_real_claude_classification():
    import asyncio
    agent = get_default_intent_agent()
    result = asyncio.run(agent.classify("I need stuff for a pasta dinner"))

    assert result.intent == Intent.search
    assert result.specificity == Specificity.specific
    assert result.language == Language.en
