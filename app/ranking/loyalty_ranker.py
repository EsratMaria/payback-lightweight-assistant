from __future__ import annotations

from typing import Optional

from app.models.schemas import Product, ProductRecommendation, UserContext


class LoyaltyRanker:
    """Re-ranks vector search results using a loyalty-aware composite score.

    Scoring formula
    ---------------
    ::

        final_score = α * semantic_similarity
                    + β * loyalty_boost
                    + γ * partner_diversity_bonus

    where:

    * **semantic_similarity** — cosine similarity returned by the vector store (0–1).

    * **loyalty_boost** — reward for high-value PAYBACK products::

          loyalty_boost = points_multiplier * (1.5 if active_promo else 1.0)

      A product with ``points_multiplier=3.0`` and an active promo scores 4.5,
      while a baseline product (multiplier=1.0, no promo) scores 1.0.

    * **partner_diversity_bonus** — steers results away from the user's dominant
      partner to surface variety.  Calculation depends on ``is_new_user``:

      - **New user** (``is_new_user=True``): ``γ`` acts as a flat diversity weight.
        Partners under-represented in the current result set receive a bonus so that
        the top-N results span all three partners, preventing cold-start monopoly.

      - **Returning user**: ``partner_diversity_bonus = 1 - partner_affinity[partner]``
        Products from partners the user interacts with *less* receive a higher bonus,
        nudging discovery without overriding the user's established preferences.

    Default weights: α=0.6, β=0.3, γ=0.1.
    The weights sum to 1.0 only when loyalty_boost == 1.0 and diversity == 1.0;
    in practice they are treated as relative contributions, not a probability
    distribution, so final_score may exceed 1.0.

    Usage
    -----
    ::

        ranker = LoyaltyRanker()
        recommendations = ranker.rank(products_with_scores, user_context)
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    async def rank(
        self,
        products_with_scores: list[tuple[Product, float]],
        user_context: Optional[UserContext],
    ) -> list[ProductRecommendation]:
        """STUB for Step 5. Real loyalty-weighted ranking lands in Step 6.

        For now: pass through retrieval results in order, wrap as
        ProductRecommendation with loyalty_boost=0 and final_score=semantic_score.
        """
        return [
            ProductRecommendation(
                product=product,
                semantic_score=score,
                loyalty_boost=0.0,
                final_score=score,
                rank=i + 1,
            )
            for i, (product, score) in enumerate(products_with_scores)
        ]
