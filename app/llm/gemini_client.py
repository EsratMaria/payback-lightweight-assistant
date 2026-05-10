"""Gemini-backed LLM client — production stub.

This class implements the LLMClient interface against Google's google-genai SDK,
intended as an alternative provider for production deployments on GCP.

WHY DEFER GEMINI:
    PAYBACK is on Google Cloud and uses Gemini in production. A second LLM
    provider behind the same LLMClient interface is the natural extension of
    this design. It is deferred from this submission to focus engineering
    time on the loyalty ranker, GCP deployment, and load testing — areas with
    higher marginal impact for the demo.

    The architectural signal is the LLMClient INTERFACE, not the second
    implementation. Adding GeminiClient is a single class change. Nothing
    else in the codebase needs to be modified — IntentAgent, the router, and
    every LLM-using component depends only on the abstract interface.

PRODUCTION IMPLEMENTATION SKETCH:
    Use the google-genai SDK with response_schema for structured output:

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=settings.google_api_key)
        json_schema = schema.model_json_schema()
        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=json_schema,
                max_output_tokens=max_tokens,
            ),
        )
        return schema.model_validate_json(response.text)

KNOWN GOTCHAS:
    - google-genai's response_schema does not support all JSON-schema features
      (e.g. some patterns around enum + description). Test schema compatibility
      and apply workarounds (flatten enums to strings with descriptions) where
      needed.
    - Retry logic should mirror ClaudeClient: one retry on ValidationError or
      JSON decode error, then raise LLMError.

WHEN TO ADD:
    Add this implementation in the GCP deployment phase or when running
    cost/quality A/B tests across providers. Toggle via DEFAULT_LLM_PROVIDER
    env var; no code changes elsewhere.
"""
from __future__ import annotations

from pydantic import BaseModel

from app.llm.base import LLMClient


class GeminiClient(LLMClient):
    """Google Gemini implementation of LLMClient — not yet implemented.

    See module docstring for full architecture, implementation sketch, and
    known gotchas. Set DEFAULT_LLM_PROVIDER=gemini in .env and implement
    using the google-genai SDK when ready.
    """

    async def structured_completion(
        self,
        prompt: str,
        schema: type[BaseModel],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> BaseModel:
        # TODO: implement using google-genai SDK with response_schema
        raise NotImplementedError(
            "GeminiClient is not yet implemented. "
            "See app/llm/gemini_client.py module docstring for the implementation sketch."
        )
