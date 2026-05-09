from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class GeminiClient(LLMClient):
    """Google Gemini implementation of LLMClient."""

    async def structured_completion(
        self, prompt: str, schema: type[T], **kwargs
    ) -> T:
        # TODO: implement using google-genai SDK with response_schema
        raise NotImplementedError
