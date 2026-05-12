"""Generate synthetic product catalogs for dm, edeka, and amazon via Claude.

Run:
    python data/generate_catalogs.py

Output:
    data/catalogs/dm.json      (200 products)
    data/catalogs/edeka.json   (200 products)
    data/catalogs/amazon.json  (300 products)

Requires UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC and UNIFIED_ENDPOINT_KEY in .env.

Generation strategy
-------------------
Primary:  Anthropic tool use with `tool_choice={"type":"tool"}` — the API guarantees
          a valid JSON object matching the declared input_schema, so no string parsing
          is needed.
Fallback: If the tool block is absent (proxy quirk / timeout), we strip markdown fences
          and extract the first JSON array from raw text.
Validation: Every raw item is validated through the Pydantic `Product` schema before
          being accepted. Batches with ≥80 % valid items are accepted; below that the
          batch is retried up to MAX_RETRIES times with exponential back-off.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

# Resolve .env relative to repo root regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_REPO_ROOT / ".env")

try:
    import anthropic
except ImportError:
    sys.exit("anthropic package not installed. Run: pip install anthropic")

sys.path.insert(0, str(_REPO_ROOT))
from app.models.schemas import Product

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_DIR = Path(__file__).parent / "catalogs"
CATALOG_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 25
MAX_RETRIES = 4
ACCEPT_THRESHOLD = 0.80  # accept batch if ≥80 % of items pass Pydantic validation

# ---------------------------------------------------------------------------
# Tool definition — JSON schema enforced by the Anthropic API
# ---------------------------------------------------------------------------

PRODUCT_TOOL: dict[str, Any] = {
    "name": "submit_products",
    "description": (
        "Submit the generated product batch as structured data. "
        "Call this tool exactly once with all products for the requested batch."
    ),
    "input_schema": {
        "type": "object",
        "required": ["products"],
        "properties": {
            "products": {
                "type": "array",
                "description": "The generated product list.",
                "items": {
                    "type": "object",
                    "required": [
                        "product_id",
                        "partner",
                        "name",
                        "description",
                        "category",
                        "price_eur",
                        "points_multiplier",
                        "active_promo",
                        "promo_text",
                    ],
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "Format: {partner}-NNNN (zero-padded 4 digits)",
                        },
                        "partner": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {
                            "type": "string",
                            "description": "1–2 honest sentences.",
                        },
                        "category": {"type": "string"},
                        "price_eur": {"type": "number", "exclusiveMinimum": 0},
                        "points_multiplier": {
                            "type": "number",
                            "enum": [1.0, 2.0, 3.0],
                            "description": "1.0 for most (≥70 %), 2.0 occasional, 3.0 rare",
                        },
                        "active_promo": {"type": "boolean"},
                        "promo_text": {
                            "type": ["string", "null"],
                            "description": "Non-null only when active_promo is true.",
                        },
                    },
                },
            }
        },
    },
}

# ---------------------------------------------------------------------------
# Partner specs
# ---------------------------------------------------------------------------

PARTNER_SPECS: dict[str, dict] = {
    "dm": {
        "count": 200,
        "description": "German drugstore chain (dm-drogerie markt)",
        "categories": [
            "personal_care",
            "baby",
            "cosmetics",
            "household",
            "supplements",
            "organic_food",
        ],
        "price_range": (0.49, 29.99),
        "language_note": (
            "Mix language naturally — some German names "
            '(e.g. "Bio Hafermilch", "Sonnenschutz Creme LSF 50"), '
            'some English brand names (e.g. "Nivea Soft Cream", "Pampers Active Fit"). '
            "About 60 % German, 40 % branded English."
        ),
        "brand_examples": (
            "dm Eigenmarken (alverde, babylove, balea), Nivea, Pampers, Persil, tetesept, Philips"
        ),
    },
    "edeka": {
        "count": 200,
        "description": "German supermarket chain (EDEKA)",
        "categories": [
            "fresh_produce",
            "dairy",
            "meat",
            "bakery",
            "frozen",
            "beverages",
            "pantry",
        ],
        "price_range": (0.99, 24.99),
        "language_note": (
            "CRITICAL: The vast majority of names MUST be in German "
            '(e.g. "Bio Vollmilch 1L", "Kartoffeln festkochend 2kg", '
            '"Frische Bratwurst 400g", "Vollkornbrot 500g", "Orangensaft 1L"). '
            "This is essential for the multilingual embedding test — German names "
            "must dominate so the retrieval system is exercised on real DE text. "
            "About 75 % German, 25 % branded English."
        ),
        "brand_examples": (
            "EDEKA Bio, Gut & Günstig, Bauer, Müller, Weihenstephan, regional bakery brands"
        ),
    },
    "amazon": {
        "count": 300,
        "description": "Amazon Germany marketplace",
        "categories": [
            "electronics",
            "home",
            "books",
            "toys",
            "kitchen",
            "fashion",
            "sports",
            "tools",
        ],
        "price_range": (4.99, 299.99),
        "language_note": (
            "Names mostly English with brand-specific conventions. "
            "Mix international and German product names naturally. "
            "Long-tail SKUs are welcome (accessories, cables, adapters, cases)."
        ),
        "brand_examples": (
            "Anker, Logitech, Bosch, Philips, Amazon Basics, LEGO, Tefal, WD, Samsung, Kindle"
        ),
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a data generation assistant for the PAYBACK loyalty app. "
    "Your job is to generate realistic, diverse product data for a multilingual retrieval test. "
    "Quality and variety matter — avoid repetitive names or descriptions. "
    "Always call the submit_products tool with EXACTLY the number of products requested. "
    "Never add commentary outside the tool call."
)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(partner: str, spec: dict, batch_index: int, batch_size: int) -> str:
    start_id = batch_index * batch_size + 1
    end_id = start_id + batch_size - 1
    categories_str = " | ".join(spec["categories"])
    description_lang = "German" if partner in ("dm", "edeka") else "English"

    return (
        f"Generate exactly {batch_size} products for {spec['description']}.\n\n"
        f"Product IDs: sequential from \"{partner}-{start_id:04d}\" to \"{partner}-{end_id:04d}\"\n"
        f"Partner field: always \"{partner}\"\n"
        f"Valid categories (use exact values): {categories_str}\n"
        f"Price range: €{spec['price_range'][0]} – €{spec['price_range'][1]}\n"
        f"Typical brands: {spec['brand_examples']}\n\n"
        f"Language rules: {spec['language_note']}\n\n"
        f"Additional rules:\n"
        f"- Spread products evenly across all {len(spec['categories'])} categories\n"
        f"- points_multiplier distribution: 1.0 for ≥70 %, 2.0 for ~20 %, 3.0 for ~10 %\n"
        f"- active_promo true for ~15 % of products; promo_text MUST be non-null iff active_promo is true\n"
        f"- description: 1–2 honest sentences in {description_lang}\n\n"
        f"Submit all {batch_size} products via the submit_products tool."
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_from_tool_use(message: anthropic.types.Message) -> list[dict] | None:
    """Extract products list from a tool_use content block."""
    for block in message.content:
        if block.type == "tool_use" and block.name == "submit_products":
            products = block.input.get("products")
            if isinstance(products, list):
                return products
    return None


def _extract_from_text(message: anthropic.types.Message) -> list[dict] | None:
    """Fallback: strip markdown fences and extract first JSON array from raw text."""
    text = " ".join(
        block.text for block in message.content if hasattr(block, "text")
    ).strip()
    if not text:
        return None

    # Strip ```json / ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Find the outermost [ ... ]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        result = json.loads(text[start : end + 1])
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_batch(
    raw_items: list[dict],
    partner: str,
    spec: dict,
) -> tuple[list[dict], list[str]]:
    """Validate raw dicts through the Pydantic Product schema.

    Returns (valid_product_dicts, error_summaries).
    Invalid items are skipped rather than raising, so the caller can decide
    whether to retry or accept a partial batch.
    """
    valid: list[dict] = []
    errors: list[str] = []
    valid_categories = set(spec["categories"])

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            errors.append(f"item[{idx}]: not a dict — skipped")
            continue

        # Normalise category to lowercase snake_case
        if "category" in item:
            item["category"] = str(item["category"]).lower().replace(" ", "_")

        # Enforce partner field consistency
        item["partner"] = partner

        # Enforce promo_text / active_promo consistency
        if not item.get("active_promo"):
            item["promo_text"] = None
        elif item.get("promo_text") is None:
            item["promo_text"] = "Sonderangebot!"

        if item.get("category") not in valid_categories:
            errors.append(
                f"item[{idx}] '{item.get('name', '?')}': "
                f"unknown category '{item.get('category')}' — skipped"
            )
            continue

        try:
            product = Product.model_validate(item)
            valid.append(product.model_dump())
        except ValidationError as exc:
            errors.append(
                f"item[{idx}] '{item.get('name', '?')}': "
                f"{exc.error_count()} error(s) — {exc.errors()[0]['msg']}"
            )

    return valid, errors


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def _generate_batch(
    client: anthropic.Anthropic,
    partner: str,
    spec: dict,
    batch_index: int,
    batch_size: int,
) -> list[dict]:
    """Generate and validate one batch; retry up to MAX_RETRIES on failure."""
    prompt = build_prompt(partner, spec, batch_index, batch_size)

    for attempt in range(1, MAX_RETRIES + 1):
        raw_items: list[dict] | None = None

        try:
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=16_000,
                system=SYSTEM_PROMPT,
                tools=[PRODUCT_TOOL],
                tool_choice={"type": "tool", "name": "submit_products"},
                messages=[{"role": "user", "content": prompt}],
            )

            # Primary: structured tool-use output
            raw_items = _extract_from_tool_use(message)

            # Fallback: parse raw text if tool block is missing
            if raw_items is None:
                print(f"      [attempt {attempt}] no tool block — trying text extraction")
                raw_items = _extract_from_text(message)

        except anthropic.APIStatusError as exc:
            print(f"      [attempt {attempt}] API error {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError as exc:
            print(f"      [attempt {attempt}] connection error: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"      [attempt {attempt}] unexpected error: {exc}")

        if raw_items is None:
            print(f"      [attempt {attempt}] no parseable output — retrying in {attempt * 2}s")
            time.sleep(attempt * 2)
            continue

        valid, errors = _validate_batch(raw_items, partner, spec)

        if errors:
            shown = errors[:3]
            suffix = f" (+ {len(errors) - 3} more)" if len(errors) > 3 else ""
            for msg in shown:
                print(f"        ⚠ {msg}")
            if suffix:
                print(f"        ⚠{suffix}")

        ratio = len(valid) / batch_size
        if ratio >= ACCEPT_THRESHOLD:
            if len(valid) < batch_size:
                print(
                    f"      [attempt {attempt}] accepted {len(valid)}/{batch_size} "
                    f"({ratio:.0%} ≥ {ACCEPT_THRESHOLD:.0%} threshold)"
                )
            return valid

        print(
            f"      [attempt {attempt}] {len(valid)}/{batch_size} valid "
            f"({ratio:.0%} < {ACCEPT_THRESHOLD:.0%} threshold) — retrying in {attempt * 2}s"
        )
        time.sleep(attempt * 2)

    raise RuntimeError(
        f"Batch {batch_index + 1} for '{partner}' failed after {MAX_RETRIES} attempts."
    )


# ---------------------------------------------------------------------------
# Partner-level orchestration
# ---------------------------------------------------------------------------


def generate_partner_catalog(
    client: anthropic.Anthropic, partner: str, spec: dict
) -> list[dict]:
    total = spec["count"]
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    products: list[dict] = []

    print(f"  Generating {total} products in {num_batches} batches of ≤{BATCH_SIZE}...")

    for i in range(num_batches):
        batch_size = min(BATCH_SIZE, total - i * BATCH_SIZE)
        batch = _generate_batch(client, partner, spec, i, batch_size)
        products.extend(batch)
        print(
            f"    Batch {i + 1}/{num_batches}: "
            f"{len(batch)} valid  (running total: {len(products)})"
        )
        time.sleep(0.5)

    return products


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    base_url = os.getenv("UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC")
    api_key = os.getenv("UNIFIED_ENDPOINT_KEY")
    if not base_url or not api_key:
        sys.exit(
            "UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC and UNIFIED_ENDPOINT_KEY "
            "must be set in .env."
        )

    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    for partner, spec in PARTNER_SPECS.items():
        out_path = CATALOG_DIR / f"{partner}.json"
        if out_path.exists():
            print(f"Skipping {partner}: {out_path} already exists.")
            continue

        print(f"\n{'=' * 55}")
        print(f"  Partner: {partner.upper()}  ({spec['count']} products)")
        print(f"{'=' * 55}")

        try:
            products = generate_partner_catalog(client, partner, spec)
        except RuntimeError as exc:
            print(f"\nFATAL: {exc}")
            sys.exit(1)

        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(products, fh, ensure_ascii=False, indent=2)

        print(f"\n  Saved {len(products)} products → {out_path.relative_to(_REPO_ROOT)}")

    print("\nDone. All catalogs written to data/catalogs/")


if __name__ == "__main__":
    main()