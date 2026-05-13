from __future__ import annotations

import asyncio
import shutil
import tempfile

import numpy as np
import pytest

from app.models.schemas import Partner, Product
from app.retrieval.embedder import Embedder
from app.retrieval.local_store import LocalChromaStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_store():
    tmpdir = tempfile.mkdtemp(prefix="chroma_test_")
    store = LocalChromaStore(persist_dir=tmpdir)
    yield store
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_products() -> list[Product]:
    return [
        Product(
            product_id="edeka_milk_de",
            partner=Partner.edeka,
            name="Bio Vollmilch 1L",
            description=(
                "Frische Bio-Vollmilch aus bayerischer Landwirtschaft, "
                "naturbelassen und reich an Kalzium."
            ),
            category="dairy",
            price_eur=1.29,
            points_multiplier=1.0,
            active_promo=False,
        ),
        Product(
            product_id="dm_shampoo_en",
            partner=Partner.dm,
            name="Balea Moisture Shampoo 300ml",
            description=(
                "Nourishing shampoo for dry and damaged hair with "
                "panthenol and argan oil."
            ),
            category="personal_care",
            price_eur=1.95,
            points_multiplier=2.0,
            active_promo=True,
            promo_text="Doppelte Punkte!",
        ),
        Product(
            product_id="amazon_keyboard_en",
            partner=Partner.amazon,
            name="Logitech MK270 Wireless Keyboard",
            description=(
                "Compact wireless keyboard and mouse combo with 2.4 GHz "
                "USB receiver and 24-month battery life."
            ),
            category="electronics",
            price_eur=34.99,
            points_multiplier=1.0,
            active_promo=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------


def test_embedder_dimensionality():
    embedder = Embedder()
    vectors = embedder.encode(["hello", "world"])
    assert vectors.shape == (2, embedder.dim)
    assert vectors.dtype == np.float32


def test_embedder_normalized():
    embedder = Embedder()
    vectors = embedder.encode(["hello", "welt", "Bio Vollmilch"], normalize=True)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(vectors)), atol=1e-5)


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------


def test_add_and_count(temp_store, sample_products):
    asyncio.run(temp_store.add(sample_products))
    assert asyncio.run(temp_store.count()) == len(sample_products)


def test_search_returns_products(temp_store, sample_products):
    asyncio.run(temp_store.add(sample_products))
    results = asyncio.run(temp_store.search("dairy milk", top_k=3))
    assert len(results) >= 1
    for product, score in results:
        assert isinstance(product, Product)
        assert 0.0 <= score <= 1.0


def test_partner_filter(temp_store, sample_products):
    asyncio.run(temp_store.add(sample_products))
    results = asyncio.run(
        temp_store.search("grocery food", top_k=10, partner_filter=Partner.edeka)
    )
    assert len(results) >= 1
    for product, _ in results:
        assert product.partner == Partner.edeka


def test_cross_lingual_retrieval(temp_store, sample_products):
    """English query must surface the German-language EDEKA milk product.

    This is the key regression test for our multilingual capability claim.
    paraphrase-multilingual-MiniLM-L12-v2 maps 'milk' and 'Vollmilch' to
    nearby vectors — if it doesn't, the model choice is wrong.
    """
    asyncio.run(temp_store.add(sample_products))
    results = asyncio.run(temp_store.search("milk", top_k=3))
    top_ids = [p.product_id for p, _ in results]
    assert "edeka_milk_de" in top_ids, (
        f"Expected 'edeka_milk_de' in top-3 for English query 'milk', got {top_ids}"
    )


def test_idempotent_upsert(temp_store, sample_products):
    asyncio.run(temp_store.add(sample_products))
    asyncio.run(temp_store.add(sample_products))
    assert asyncio.run(temp_store.count()) == len(sample_products)


def test_promo_only_filter_excludes_non_promo_products(temp_store):
    """promo_only=True returns only products with active_promo=True."""
    products = [
        Product(
            product_id="promo-001",
            partner=Partner.dm,
            name="Balea Shampoo Angebot",
            description="Pflegeshampoo mit Arganöl, jetzt im Angebot.",
            category="personal_care",
            price_eur=1.49,
            active_promo=True,
            promo_text="20% Rabatt!",
        ),
        Product(
            product_id="no-promo-001",
            partner=Partner.dm,
            name="Balea Duschgel",
            description="Pflegendes Duschgel mit Sheabutter.",
            category="personal_care",
            price_eur=1.29,
            active_promo=False,
        ),
        Product(
            product_id="no-promo-002",
            partner=Partner.dm,
            name="Balea Bodymilk",
            description="Reichhaltige Bodymilk für trockene Haut.",
            category="personal_care",
            price_eur=1.99,
            active_promo=False,
        ),
    ]
    asyncio.run(temp_store.add(products))

    promo_results = asyncio.run(temp_store.search("Balea Pflegeprodukt", top_k=10, promo_only=True))
    assert len(promo_results) == 1
    assert promo_results[0][0].product_id == "promo-001"
    assert promo_results[0][0].active_promo is True

    all_results = asyncio.run(temp_store.search("Balea Pflegeprodukt", top_k=10, promo_only=False))
    assert len(all_results) == 3
