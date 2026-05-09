"""Simple load test for the /assist endpoint using httpx async.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --rps 10 --duration 30
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

QUERIES = [
    "Shampoo für trockenes Haar",
    "dm bio produkte",
    "I need a birthday gift for my dad",
    "Welche Produkte gibt es von Edeka?",
    "günstige Waschmittel",
    "amazon elektronik",
    "Vitamin D Tabletten",
    "organic food edeka",
    "Zahnbürste elektrisch",
    "Kaffee Angebote",
]


async def send_request(client: httpx.AsyncClient, url: str) -> float:
    query = QUERIES[int(time.time() * 1000) % len(QUERIES)]
    start = time.perf_counter()
    resp = await client.post(
        f"{url}/assist",
        json={"query": query},
        timeout=10.0,
    )
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000


async def run(url: str, rps: int, duration: int) -> None:
    latencies: list[float] = []
    errors = 0
    interval = 1.0 / rps
    deadline = time.perf_counter() + duration

    async with httpx.AsyncClient() as client:
        while time.perf_counter() < deadline:
            task = asyncio.create_task(send_request(client, url))
            await asyncio.sleep(interval)
            try:
                latencies.append(await task)
            except Exception as e:
                errors += 1
                print(f"Error: {e}")

    if latencies:
        print(f"\nRequests: {len(latencies)}  Errors: {errors}")
        print(f"P50: {statistics.median(latencies):.1f}ms")
        print(f"P95: {sorted(latencies)[int(len(latencies) * 0.95)]:.1f}ms")
        print(f"P99: {sorted(latencies)[int(len(latencies) * 0.99)]:.1f}ms")
        print(f"Max: {max(latencies):.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps", type=int, default=5)
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.rps, args.duration))


if __name__ == "__main__":
    main()
