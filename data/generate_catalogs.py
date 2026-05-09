"""Generate synthetic product catalogs for dm, edeka, and amazon via Claude.

Run:
    python data/generate_catalogs.py

Output:
    data/catalogs/dm.json      (200 products)
    data/catalogs/edeka.json   (200 products)
    data/catalogs/amazon.json  (300 products)

Requires ANTHROPIC_API_KEY in .env.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
except ImportError:
    sys.exit("anthropic package not installed. Run: pip install anthropic")

CATALOG_DIR = Path(__file__).parent / "catalogs"
CATALOG_DIR.mkdir(exist_ok=True)

PARTNER_SPECS: dict[str, dict] = {
    "dm": {
        "count": 200,
        "description": "German drugstore chain (dm-drogerie markt)",
        "categories": [
            "Haarpflege",
            "Körperpflege",
            "Gesichtspflege",
            "Sonnenschutz",
            "Babyprodukte",
            "Vitamine & Nahrungsergänzung",
            "Haushaltsreiniger",
            "Bio-Lebensmittel",
            "Zahnpflege",
            "Deodorant",
        ],
        "price_range": (0.99, 29.99),
    },
    "edeka": {
        "count": 200,
        "description": "German supermarket chain (EDEKA)",
        "categories": [
            "Obst & Gemüse",
            "Molkereiprodukte",
            "Fleisch & Wurst",
            "Backwaren",
            "Tiefkühlkost",
            "Getränke",
            "Snacks & Süßwaren",
            "Frühstück & Cerealien",
            "Konserven & Fertiggerichte",
            "Bio & Vegan",
        ],
        "price_range": (0.49, 49.99),
    },
    "amazon": {
        "count": 300,
        "description": "Amazon Germany marketplace",
        "categories": [
            "Elektronik",
            "Bücher",
            "Spielzeug",
            "Sport & Freizeit",
            "Küche & Haushalt",
            "Bekleidung",
            "Garten & Terrasse",
            "Bürobedarf",
            "Heimwerker",
            "Musik & Film",
            "Lebensmittel & Gourmet",
            "Beauty",
        ],
        "price_range": (4.99, 499.99),
    },
}

BATCH_SIZE = 25


def build_prompt(partner: str, spec: dict, batch_index: int, batch_size: int) -> str:
    start_id = batch_index * batch_size + 1
    return f"""You are generating realistic synthetic product data for the PAYBACK loyalty app.

Partner: {spec["description"]}
Categories available: {", ".join(spec["categories"])}
Price range: €{spec["price_range"][0]} – €{spec["price_range"][1]}

Generate exactly {batch_size} products starting at product_id "{partner}-{start_id:04d}".
Each product must be realistic, varied across categories, and plausible for this retailer.

Return ONLY a valid JSON array of objects with these exact fields:
- product_id: string, format "{partner}-NNNN" (zero-padded 4 digits, sequential from {start_id})
- partner: "{partner}"
- name: string (realistic German or English product name)
- description: string (1–2 sentence description in German)
- category: string (one of the categories listed above)
- price_eur: number (realistic price within the range)
- points_multiplier: number (1.0, 1.5, 2.0, or 3.0 — weight toward 1.0)
- active_promo: boolean (true for ~15% of products)
- promo_text: string or null (non-null only when active_promo is true, e.g. "Doppelte Punkte!")

No markdown, no commentary — raw JSON array only."""


def generate_partner_catalog(
    client: anthropic.Anthropic, partner: str, spec: dict
) -> list[dict]:
    total = spec["count"]
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    products: list[dict] = []

    print(f"  Generating {total} products for {partner} in {num_batches} batches...")

    for i in range(num_batches):
        current_batch_size = min(BATCH_SIZE, total - i * BATCH_SIZE)
        prompt = build_prompt(partner, spec, i, current_batch_size)

        for attempt in range(3):
            try:
                message = client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = message.content[0].text.strip()
                batch = json.loads(raw)
                if not isinstance(batch, list):
                    raise ValueError("Expected a JSON array")
                products.extend(batch)
                print(f"    Batch {i + 1}/{num_batches}: {len(batch)} products OK")
                time.sleep(0.3)
                break
            except (json.JSONDecodeError, ValueError) as e:
                print(f"    Batch {i + 1} attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(1.0)

    return products


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    client = anthropic.Anthropic(api_key=api_key)

    for partner, spec in PARTNER_SPECS.items():
        out_path = CATALOG_DIR / f"{partner}.json"
        if out_path.exists():
            print(f"Skipping {partner}: {out_path} already exists.")
            continue

        print(f"\nGenerating catalog: {partner}")
        products = generate_partner_catalog(client, partner, spec)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2)

        print(f"  Saved {len(products)} products to {out_path}")

    print("\nDone. Catalogs written to data/catalogs/")


if __name__ == "__main__":
    main()
