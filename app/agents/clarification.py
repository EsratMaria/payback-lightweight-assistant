"""Clarification agent.

Invoked when the IntentResult has specificity == "vague" and the router decides
that asking a follow-up question will yield a better result than guessing.

Responsibilities:
- Generate a single, concise clarifying question in the same language as the
  original query (EN or DE).
- Produce 3–5 short suggested_options (quick replies) so the user can tap-to-answer
  on mobile without typing.
- Use the LLM to generate the question, seeded with the extracted_query and the
  list of known partners / categories as context.

The output is a ClarifyingQuestion pydantic model, ready to embed in AssistantResponse.
"""
from __future__ import annotations

from app.config import settings
from app.llm.base import LLMClient
from app.models.schemas import ClarifyingQuestion, Language, Product

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLARIFICATION_SYSTEM_PROMPT = (
    "You generate catalog-grounded clarifying questions for vague user queries on "
    "PAYBACK's shopping assistant. The user's query was too vague — or the catalog "
    "had no strong matches — to retrieve confidently.\n\n"
    "Your job:\n"
    "- Look at what the catalog actually returned (or note when it returned weak results).\n"
    "- Ask the user ONE concise clarifying question that helps narrow their intent.\n"
    "- Provide 2-4 suggested options. Ground them in what's actually available in the "
    "catalog — do not invent categories that don't appear in the results.\n"
    "- If catalog results are weak (no strong matches), ask a more general clarifying "
    "question without inventing specifics.\n"
    "- Match the user's language (German or English).\n\n"
    "Return a ClarifyingQuestion object via the `respond` tool."
)

CLARIFICATION_USER_PROMPT_TEMPLATE = (
    'The user asked: "{query}"\n\n'
    "Language: {language}\n"
    "Catalog match quality: {match_quality}\n\n"
    "Top catalog results:\n"
    "{formatted_results}\n\n"
    "Generate ONE clarifying question with 2-4 grounded suggested options. "
    "Match the user's language."
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ClarificationAgent:
    """Generates catalog-grounded clarifying questions for vague queries.

    Provider-agnostic: any LLMClient implementation can be injected.
    The router calls this after retrieval, passing the top results so the
    question can reference what is actually in the catalog.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def generate(
        self,
        query: str,
        top_results: list[tuple[Product, float]],
        language: Language,
    ) -> ClarifyingQuestion:
        """Generate a clarifying question grounded in the catalog results."""
        weak = not top_results or all(
            score < settings.retrieval_weak_threshold
            for _, score in top_results
        )
        match_quality = "weak" if weak else "strong"

        if top_results:
            lines = [
                f"  {i}. [{p.partner.value}] {p.name} ({p.category}, score={score:.2f})"
                for i, (p, score) in enumerate(top_results[:10], 1)
            ]
            formatted_results = "\n".join(lines)
        else:
            formatted_results = (
                "No results returned. Generate a generic clarifying question "
                "for a vague shopping query."
            )

        user_prompt = CLARIFICATION_USER_PROMPT_TEMPLATE.format(
            query=query,
            language=language.value,
            match_quality=match_quality,
            formatted_results=formatted_results,
        )

        result = await self.llm.structured_completion(
            prompt=user_prompt,
            schema=ClarifyingQuestion,
            system=CLARIFICATION_SYSTEM_PROMPT,
            max_tokens=512,
        )
        return result  # type: ignore[return-value]
