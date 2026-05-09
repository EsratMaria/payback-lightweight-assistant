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
