from __future__ import annotations

import pytest

from app.models.schemas import Partner, Product, UserContext
from app.ranking.loyalty_ranker import LoyaltyRanker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _product(
    partner: Partner,
    product_id: str = "p1",
    points_multiplier: float = 1.0,
    active_promo: bool = False,
) -> Product:
    return Product(
        product_id=product_id,
        partner=partner,
        name=f"Product {product_id}",
        description="Test product",
        category="test",
        price_eur=5.0,
        points_multiplier=points_multiplier,
        active_promo=active_promo,
    )


def _known_user(
    edeka: float = 0.7,
    dm: float = 0.2,
    amazon: float = 0.1,
) -> UserContext:
    return UserContext(
        user_id="u1",
        partner_affinity={
            Partner.edeka: edeka,
            Partner.dm: dm,
            Partner.amazon: amazon,
        },
        is_new_user=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_ranking_known_user(monkeypatch):
    """Known EDEKA-heavy user: Amazon and dm get higher diversity bonus → rank above EDEKA."""
    import app.config as cfg
    monkeypatch.setattr(cfg.settings, "loyalty_weight_semantic", 0.6)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_commercial", 0.3)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_diversity", 0.1)
    monkeypatch.setattr(cfg.settings, "affinity_diversity_scale", 0.5)
    monkeypatch.setattr(cfg.settings, "points_multiplier_scale", 0.4)
    monkeypatch.setattr(cfg.settings, "active_promo_bonus", 0.3)

    products = [
        (_product(Partner.edeka, "edeka-1"), 0.7),
        (_product(Partner.dm, "dm-1"), 0.7),
        (_product(Partner.amazon, "amazon-1"), 0.7),
    ]
    ranker = LoyaltyRanker()
    results = await ranker.rank(products, _known_user())

    assert results[0].rank == 1
    assert results[0].product.partner == Partner.amazon


@pytest.mark.asyncio
async def test_commercial_boost_lifts_low_match(monkeypatch):
    """Product B with 2x points + promo should outscore product A despite lower semantic match."""
    import app.config as cfg
    monkeypatch.setattr(cfg.settings, "loyalty_weight_semantic", 0.6)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_commercial", 0.3)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_diversity", 0.1)
    monkeypatch.setattr(cfg.settings, "affinity_diversity_scale", 0.5)
    monkeypatch.setattr(cfg.settings, "points_multiplier_scale", 0.4)
    monkeypatch.setattr(cfg.settings, "active_promo_bonus", 0.3)

    products = [
        (_product(Partner.dm, "a", points_multiplier=1.0, active_promo=False), 0.85),
        (_product(Partner.dm, "b", points_multiplier=2.0, active_promo=True), 0.70),
    ]
    ranker = LoyaltyRanker()
    results = await ranker.rank(products, user_context=None)

    # Product B: commercial = (2-1)*0.4 + 0.3 = 0.7, diversity = 1 - 2/2 = 0.0
    # final_B = 0.6*0.70 + 0.3*0.7 + 0.1*0.0 = 0.42 + 0.21 = 0.63
    # Product A: commercial = 0.0, diversity = 0.0
    # final_A = 0.6*0.85 + 0.0 + 0.0 = 0.51
    assert results[0].product.product_id == "b"
    assert abs(results[0].final_score - 0.63) < 0.001
    assert abs(results[1].final_score - 0.51) < 0.001


@pytest.mark.asyncio
async def test_cold_start_falls_back_to_set_diversity(monkeypatch):
    """Cold-start (user_context=None): minority partner (dm) surfaces above majority (edeka)."""
    import app.config as cfg
    monkeypatch.setattr(cfg.settings, "loyalty_weight_semantic", 0.6)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_commercial", 0.3)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_diversity", 0.1)
    monkeypatch.setattr(cfg.settings, "affinity_diversity_scale", 0.5)
    monkeypatch.setattr(cfg.settings, "points_multiplier_scale", 0.4)
    monkeypatch.setattr(cfg.settings, "active_promo_bonus", 0.3)

    products = [
        (_product(Partner.edeka, f"e{i}"), 0.7) for i in range(4)
    ] + [
        (_product(Partner.dm, "dm-1"), 0.7),
    ]
    ranker = LoyaltyRanker()
    results = await ranker.rank(products, user_context=None)

    # dm diversity = 1 - 1/5 = 0.8 → final = 0.42 + 0.08 = 0.50
    # edeka diversity = 1 - 4/5 = 0.2 → final = 0.42 + 0.02 = 0.44
    assert results[0].product.partner == Partner.dm


@pytest.mark.asyncio
async def test_is_new_user_treated_as_cold_start(monkeypatch):
    """is_new_user=True triggers the same set-diversity path as user_context=None."""
    import app.config as cfg
    monkeypatch.setattr(cfg.settings, "loyalty_weight_semantic", 0.6)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_commercial", 0.3)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_diversity", 0.1)
    monkeypatch.setattr(cfg.settings, "affinity_diversity_scale", 0.5)
    monkeypatch.setattr(cfg.settings, "points_multiplier_scale", 0.4)
    monkeypatch.setattr(cfg.settings, "active_promo_bonus", 0.3)

    products = [
        (_product(Partner.edeka, f"e{i}"), 0.7) for i in range(4)
    ] + [
        (_product(Partner.dm, "dm-1"), 0.7),
    ]
    new_user = UserContext(
        user_id="new",
        partner_affinity={Partner.edeka: 0.7, Partner.dm: 0.2, Partner.amazon: 0.1},
        is_new_user=True,
    )
    ranker = LoyaltyRanker()
    results = await ranker.rank(products, user_context=new_user)

    assert results[0].product.partner == Partner.dm


@pytest.mark.asyncio
async def test_weights_configurable(monkeypatch):
    """Setting weight_semantic=1.0, others=0.0 makes ranking purely semantic."""
    import app.config as cfg
    monkeypatch.setattr(cfg.settings, "loyalty_weight_semantic", 1.0)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_commercial", 0.0)
    monkeypatch.setattr(cfg.settings, "loyalty_weight_diversity", 0.0)
    monkeypatch.setattr(cfg.settings, "affinity_diversity_scale", 0.5)
    monkeypatch.setattr(cfg.settings, "points_multiplier_scale", 0.4)
    monkeypatch.setattr(cfg.settings, "active_promo_bonus", 0.3)

    products = [
        (_product(Partner.edeka, "low"), 0.5),
        (_product(Partner.dm, "high"), 0.9),
        (_product(Partner.amazon, "mid"), 0.7),
    ]
    ranker = LoyaltyRanker()
    results = await ranker.rank(products, user_context=None)

    assert [r.product.product_id for r in results] == ["high", "mid", "low"]
