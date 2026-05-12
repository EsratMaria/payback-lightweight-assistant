"""Smoke-test the live API.

Start the server first: uvicorn app.main:app --reload
Then run: python scripts/try_api.py
"""
from __future__ import annotations

import asyncio

import httpx

BASE_URL = "http://localhost:8000"


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:

        # 1. Health check
        r = await client.get("/health")
        print(f"GET /health → {r.status_code} {r.json()}")

        # 2. List users
        r = await client.get("/users")
        print(f"\nGET /users → {r.status_code}")
        for u in r.json():
            print(f"  {u['user_id']}: {u['description']}")

        # 3. Partners
        r = await client.get("/partners")
        print(f"\nGET /partners → {r.status_code}")
        for p in r.json():
            print(f"  [{p['partner']}] {p['name']} — {p['product_count']} products")

        # 4. Stats
        r = await client.get("/stats")
        print(f"\nGET /stats → {r.status_code}")
        s = r.json()
        print(f"  total_products={s['total_products']}")
        print(f"  per_partner={s['products_per_partner']}")
        print(f"  embedding_model={s['embedding_model']}")
        print(f"  vector_store={s['vector_store_backend']}")

        # 5. POST /assist — cold start
        print("\n--- POST /assist (cold start, no user_id) ---")
        r = await client.post("/assist", json={"query": "stuff for a pasta dinner"})
        body = r.json()
        _print_response(body)

        # 6. POST /assist — EDEKA-heavy user (should surface Amazon/dm higher)
        print("\n--- POST /assist (user_edeka_heavy) ---")
        r = await client.post(
            "/assist",
            json={"query": "stuff for a pasta dinner", "user_id": "user_edeka_heavy"},
        )
        body = r.json()
        _print_response(body)

        # 7. POST /assist — German query
        print("\n--- POST /assist (German) ---")
        r = await client.post("/assist", json={"query": "günstige Windeln"})
        body = r.json()
        _print_response(body)

        # 8. POST /assist — unknown user_id (should soft-fail, not 404)
        print("\n--- POST /assist (unknown user_id) ---")
        r = await client.post(
            "/assist",
            json={"query": "wireless mouse", "user_id": "ghost_user_123"},
        )
        print(f"  status_code={r.status_code} (expected 200, not 404)")
        _print_response(r.json())


def _print_response(body: dict) -> None:
    print(
        f"  response_type={body['response_type']} | "
        f"intent={body['intent_result']['intent']} | "
        f"latency={body['latency_ms']:.0f}ms | "
        f"cost=€{body['estimated_cost_eur']:.5f}"
    )
    if body["response_type"] == "recommendations":
        if body.get("debug_expanded_queries"):
            print(f"  expanded_queries: {body['debug_expanded_queries']}")
        for rec in body["recommendations"][:3]:
            p = rec["product"]
            print(
                f"    {rec['rank']}. [{p['partner']}] {p['name']} "
                f"(final={rec['final_score']:.2f})"
            )
    elif body["response_type"] == "clarification":
        print(f"  question: {body['clarification']['question']}")
        print(f"  options:  {body['clarification']['suggested_options']}")
    elif body["response_type"] == "navigation":
        print(f"  navigate to: {body['navigation_target']}")


if __name__ == "__main__":
    asyncio.run(main())
