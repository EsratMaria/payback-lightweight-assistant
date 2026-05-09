"""Query router.

Decides which action branch to take after intent classification:

  - "recommendations": specificity is "specific" or intent is "search" / "discovery"
    with confidence > 0.7 → run retrieval + ranking, return ProductRecommendations.

  - "clarification": specificity is "vague" OR confidence < 0.7 → invoke the
    clarification agent, return a ClarifyingQuestion.

  - "navigation": specificity is "navigational" → resolve the target_partner or
    known app route and return a navigation_target string (deep-link URL or route name).

The router is intentionally stateless; all context flows through IntentResult and
UserContext so there is no session state to manage.
"""
