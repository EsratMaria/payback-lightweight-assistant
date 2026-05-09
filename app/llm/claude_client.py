from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class ClaudeClient(LLMClient):
    """Anthropic Claude implementation of LLMClient."""

    async def structured_completion(
        self, prompt: str, schema: type[T], **kwargs
    ) -> T:
        # TODO: implement using anthropic SDK with tool_use / json mode
        raise NotImplementedError
