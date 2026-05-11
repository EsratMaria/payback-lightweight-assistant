from __future__ import annotations

import logging

import anthropic
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.llm.base import LLMClient, LLMError

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 2


class ClaudeClient(LLMClient):
    """Anthropic Claude implementation of LLMClient.

    WHY TOOL-USE INSTEAD OF ASKING FOR JSON:
        When you ask Claude to "reply in JSON", the output is free text that happens
        to contain JSON. It may be wrapped in ```json fences, prefixed with
        "Sure, here is your answer:", or subtly malformed. Parsing it requires regex
        stripping, json.loads, and hope.

        Tool-use with `tool_choice={"type":"tool","name":"respond"}` is fundamentally
        different: the API physically cannot return free text. It is forced to call
        the tool, and the tool input is validated against the declared input_schema
        before it ever reaches our code. The result is a plain dict — no parsing
        needed, no JSONDecodeError possible at the extraction step.

        Pydantic validation still runs on our side as a second defence, because the
        API schema validation is structural (types, required fields) but does not
        enforce business-logic constraints (e.g. confidence in [0, 1]).

    RETRY POLICY:
        One retry on ValidationError or missing tool_use block. All other exceptions
        (network, auth, rate-limit) propagate unchanged — the caller decides whether
        to retry at a higher level.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "claude-sonnet-4-6",
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
                        "Claude returned no tool_use block — structured output failed."
                    )
                    if attempt < _MAX_ATTEMPTS:
                        logger.warning(
                            "Claude returned no tool_use block, retrying. Schema=%s",
                            schema.__name__,
                        )
                    raise err

                return schema.model_validate(tool_input)

            except (ValidationError, LLMError) as exc:
                last_error = exc
                if attempt < _MAX_ATTEMPTS:
                    if isinstance(exc, ValidationError):
                        logger.warning(
                            "Claude structured output failed validation, retrying. "
                            "Schema=%s errors=%s raw_input=%s",
                            schema.__name__, exc.errors(), tool_input,
                        )
                    else:
                        logger.warning(
                            "Claude structured output error on attempt %d, retrying. "
                            "Schema=%s error=%s",
                            attempt, schema.__name__, exc,
                        )

        raise LLMError(
            f"Claude structured output failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )
