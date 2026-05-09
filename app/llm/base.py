from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(ABC):
    """Common interface for all LLM provider clients.

    Concrete implementations (ClaudeClient, GeminiClient) must override
    `structured_completion`. Callers pass the desired output Pydantic schema class;
    the method is responsible for prompting the model, parsing its output, and
    returning a fully validated instance of that schema.
    """

    @abstractmethod
    async def structured_completion(
        self, prompt: str, schema: type[T], **kwargs
    ) -> T:
        """Send `prompt` to the underlying LLM and return a validated Pydantic instance.

        Args:
            prompt: The full prompt string, including any system instructions and
                    the user query.
            schema: A Pydantic BaseModel subclass that defines the expected output
                    shape.  Implementations should use JSON-mode or tool-use to
                    coerce the model output into this schema before returning.
            **kwargs: Provider-specific overrides (e.g., temperature, max_tokens,
                      model version).

        Returns:
            A fully-validated instance of `schema`.

        Raises:
            ValueError: If the model output cannot be coerced into `schema`.
            httpx.HTTPStatusError: On network or provider API errors.
        """
        ...
