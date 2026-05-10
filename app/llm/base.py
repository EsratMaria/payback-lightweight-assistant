from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMError(Exception):
    """Raised when an LLM client fails to produce a valid response after exhausting retries.

    Callers catch this to surface a graceful degradation rather than an unhandled 500.
    """


class LLMClient(ABC):
    """Contract: give a prompt and a Pydantic schema, get back a validated instance.

    The caller never knows — or cares — which LLM provider is underneath. Adding a
    new provider means a new class that satisfies this interface. Nothing else in the
    codebase changes: IntentAgent, the router, and every LLM-using component depend
    only on LLMClient, never on ClaudeClient or GeminiClient directly.

    Implementations MUST:
    - Force structured output (tool-use, response_schema, or equivalent mechanism)
      so the response is guaranteed to be parseable — no free-text wrapping, no
      markdown fences, no "Sure, here is your JSON:" preambles.
    - Validate the parsed output with Pydantic before returning.
    - Retry exactly once on ValidationError or structured-output failure before
      raising LLMError. Do NOT retry on network/auth/rate-limit errors — those
      should bubble up unchanged so the caller can decide.
    """

    @abstractmethod
    async def structured_completion(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> BaseModel:
        """Call the underlying LLM and return a Pydantic-validated instance of `schema`.

        Args:
            prompt: User-turn content — the question, task, or data to classify.
            schema: Pydantic BaseModel subclass defining the expected output shape.
            system: Optional system-turn instruction. Omitted from the API call when
                    None (not passed as empty string).
            max_tokens: Upper bound on output tokens. Intent classification needs
                        ~200 tokens; generation tasks may need more.

        Returns:
            A fully-validated instance of `schema`.

        Raises:
            LLMError: After all retries are exhausted.
            anthropic.APIError / httpx error: On network or auth failures — NOT wrapped,
                so the caller can distinguish transient vs permanent errors.
        """
        ...
