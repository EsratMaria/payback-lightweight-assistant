"""Intent classification agent.

Sends the raw user query to the configured LLM and returns an IntentResult.

Responsibilities:
- Detect language (EN / DE) without a separate call — done inline in the prompt.
- Classify intent into one of: search, discovery, comparison, support.
- Determine specificity: specific (user knows what they want), vague (needs help),
  navigational (wants to go somewhere in the app).
- Extract a clean, retrieval-ready version of the query.
- Identify an explicit target partner, if any.
- Return confidence score and a short reasoning trace for observability.

The prompt is bilingual so the model doesn't need to translate before classifying.
"""
