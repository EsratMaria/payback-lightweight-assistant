from __future__ import annotations

import logging
from typing import Optional

from app.config import settings
from app.models.schemas import Partner, Product, ProductRecommendation, UserContext

logger = logging.getLogger(__name__)


class LoyaltyRanker:
    """Three-signal weighted ranker for product recommendations.

    final_score = α · semantic_similarity
                + β · commercial_loyalty_boost
                + γ · partner_diversity_bonus

    where:
      α = loyalty_weight_semantic     (default 0.6) — pure relevance to the query
      β = loyalty_weight_commercial   (default 0.3) — points multiplier + active promo
      γ = loyalty_weight_diversity    (default 0.1) — partner diversity signal

    Component formulas
    ------------------
    commercial_loyalty_boost = (points_multiplier - 1.0) * points_multiplier_scale
                              + active_promo_bonus if active_promo else 0
      — gives 0.0 for normal products (1x points, no promo)
      — gives ~0.7 for a 2x-points-with-promo product

    partner_diversity_bonus:
      KNOWN USER (user_context provided, is_new_user=False):
          (1.0 - user_context.partner_affinity[product.partner]) * affinity_diversity_scale
        surfaces partners the user shops LESS — supports PAYBACK's loyalty growth goal.

      COLD START (user_context is None OR is_new_user=True):
          1.0 - (count_of_this_partner_in_topk / topk)
        surfaces a balanced partner mix from the retrieval result set itself.

    All component scores are kept in [0, 1] so final_score also lives in [0, 1].
    Weights are read from settings on each call — change them via env vars without
    restarting the service.

    Usage
    -----
        ranker = LoyaltyRanker()
        recommendations = await ranker.rank(products_with_scores, user_context)
    """

    async def rank(
        self,
        products_with_scores: list[tuple[Product, float]],
        user_context: Optional[UserContext],
    ) -> list[ProductRecommendation]:
        """Score and rank products using semantic, commercial, and diversity signals."""
        if not products_with_scores:
            return []

        cold_start = user_context is None or user_context.is_new_user
        topk = len(products_with_scores)

        partner_counts: dict[Partner, int] = {}
        if cold_start:
            for product, _ in products_with_scores:
                partner_counts[product.partner] = partner_counts.get(product.partner, 0) + 1

        # Compute all scores first, then sort, then assign ranks
        raw: list[tuple[Product, float, float, float, float]] = []  # product, sem, commercial, diversity, final
        for product, semantic_score in products_with_scores:
            commercial = (
                (product.points_multiplier - 1.0) * settings.points_multiplier_scale
                + (settings.active_promo_bonus if product.active_promo else 0.0)
            )

            if cold_start:
                diversity = 1.0 - (partner_counts[product.partner] / topk)
            else:
                affinity = user_context.partner_affinity.get(product.partner, 0.0)
                diversity = (1.0 - affinity) * settings.affinity_diversity_scale

            final = (
                settings.loyalty_weight_semantic * semantic_score
                + settings.loyalty_weight_commercial * commercial
                + settings.loyalty_weight_diversity * diversity
            )
            raw.append((product, semantic_score, commercial, diversity, final))

        raw.sort(key=lambda x: -x[4])
        results = [
            ProductRecommendation(
                product=product,
                semantic_score=semantic,
                loyalty_boost=commercial,
                diversity_bonus=diversity,
                final_score=final,
                rank=i + 1,
            )
            for i, (product, semantic, commercial, diversity, final) in enumerate(raw)
        ]

        logger.info(
            "Ranked %d products (mode=%s, top final_score=%.3f)",
            len(results),
            "cold_start" if cold_start else "known_user",
            results[0].final_score if results else 0.0,
        )
        return results
