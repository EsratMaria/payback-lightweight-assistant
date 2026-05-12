"""Claude Opus judge client for LLM-as-judge evaluation.

Deliberately uses a different model (Opus) than the production system (Sonnet)
to reduce self-preference bias in evaluation.
"""
from __future__ import annotations

import logging

import anthropic
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.llm.base import LLMClient, LLMError

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2


class OpusJudgeClient(LLMClient):
    """Claude Opus-based judge for E2E eval.

    Deliberately a different model from the production system (which uses Sonnet).
    This reduces — though doesn't eliminate — self-preference bias in LLM-as-judge
    evaluation. In a fuller production setup we'd also cross-evaluate against a
    different provider (e.g. Gemini, GPT-4); within a single-provider constraint
    the Sonnet/Opus split is the cleanest available approach.

    The `temperature` parameter is not supported by Opus 4 models and is omitted.
    Tool-use with forced `tool_choice` already produces highly deterministic output.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "claude-opus-4-7",
    ) -> None:
        self.model = model
        resolved_key = api_key or settings.unified_endpoint_key
        resolved_url = base_url or settings.unified_endpoint_base_url_anthropic or None
        self._client = anthropic.AsyncAnthropic(
            api_key=resolved_key,
            base_url=resolved_url,
        )

    async def structured_completion(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> BaseModel:
        tool = {
            "name": "respond",
            "description": (
                f"Respond with a structured answer matching the {schema.__name__} schema."
            ),
            "input_schema": schema.model_json_schema(),
        }

        api_kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "respond"},
        }
        if system:
            api_kwargs["system"] = system

        last_error: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._client.messages.create(**api_kwargs)

                tool_input: dict | None = None
                for block in response.content:
                    if block.type == "tool_use":
                        tool_input = block.input
                        break

                if tool_input is None:
                    err = LLMError(
                        "Opus judge returned no tool_use block — structured output failed."
                    )
                    if attempt < _MAX_ATTEMPTS:
                        logger.warning(
                            "Opus judge returned no tool_use block, retrying. Schema=%s",
                            schema.__name__,
                        )
                    raise err

                return schema.model_validate(tool_input)

            except (ValidationError, LLMError) as exc:
                last_error = exc
                if attempt < _MAX_ATTEMPTS:
                    if isinstance(exc, ValidationError):
                        logger.warning(
                            "Opus judge structured output failed validation, retrying. "
                            "Schema=%s errors=%s",
                            schema.__name__, exc.errors(),
                        )
                    else:
                        logger.warning(
                            "Opus judge error on attempt %d, retrying. Schema=%s error=%s",
                            attempt, schema.__name__, exc,
                        )

        raise LLMError(
            f"Opus judge failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )
