"""Ingestion pipeline: reads catalog JSON files and loads them into the vector store.

Usage:
    python -m app.retrieval.ingest              # upsert / update existing store
    python -m app.retrieval.ingest --rebuild    # wipe and rebuild from scratch
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.models.schemas import Product
from app.retrieval.local_store import LocalChromaStore

_CATALOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "catalogs"


async def ingest(rebuild: bool = False) -> None:
    store = LocalChromaStore()

    if rebuild:
        print("Rebuilding store — deleting existing collection...")
        await store.reset()

    catalog_files = sorted(_CATALOG_DIR.glob("*.json"))
    if not catalog_files:
        print(f"No catalog files found in {_CATALOG_DIR}")
        return

    total = 0
    for path in catalog_files:
        partner = path.stem
        raw_items = json.loads(path.read_text(encoding="utf-8"))
        products = [Product.model_validate(item) for item in raw_items]
        await store.add(products)
        print(f"  {partner:<10}  {len(products):>5} products ingested")
        total += len(products)

    final_count = await store.count()
    print(f"\n  Total ingested : {total}")
    print(f"  Store count    : {final_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest product catalogs into the local ChromaDB vector store."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the existing collection before ingesting.",
    )
    args = parser.parse_args()
    asyncio.run(ingest(rebuild=args.rebuild))


if __name__ == "__main__":
    main()
