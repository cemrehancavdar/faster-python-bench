"""Generate 100K synthetic user events as a JSON file.

Deterministic (seeded) so benchmarks are reproducible.
Run: uv run data/generate.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 42
NUM_EVENTS = 100_000
NUM_USERS = 5_000
OUTPUT_PATH = Path(__file__).parent / "events.json"

EVENT_TYPES = [
    "page_view",
    "click",
    "purchase",
    "signup",
    "logout",
    "search",
    "add_to_cart",
    "remove_from_cart",
    "checkout_start",
    "error",
]

PAGES = [
    "/home",
    "/products",
    "/products/123",
    "/cart",
    "/checkout",
    "/account",
    "/search",
    "/about",
    "/contact",
    "/blog",
]

SEARCH_TERMS = [
    "python",
    "laptop",
    "headphones",
    "keyboard",
    "monitor",
    "mouse",
    "charger",
    "backpack",
    "shoes",
    "coffee",
]

# Date range: 2025-01-01 to 2025-12-31
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
DATE_RANGE_SECONDS = int((END_DATE - START_DATE).total_seconds())


def generate_metadata(rng: random.Random, event_type: str) -> dict[str, object]:
    """Generate event-type-specific metadata."""
    meta: dict[str, object] = {"page": rng.choice(PAGES)}

    if event_type == "purchase":
        meta["amount"] = round(rng.uniform(1.0, 500.0), 2)
        meta["currency"] = "USD"
        meta["item_count"] = rng.randint(1, 10)
    elif event_type == "search":
        meta["query"] = rng.choice(SEARCH_TERMS)
        meta["results_count"] = rng.randint(0, 200)
    elif event_type == "click":
        meta["element_id"] = f"btn-{rng.randint(1, 50)}"
    elif event_type == "error":
        meta["error_code"] = rng.choice([400, 404, 500, 502, 503])
        meta["message"] = f"Error on {meta['page']}"
    elif event_type == "add_to_cart":
        meta["product_id"] = rng.randint(1000, 9999)
        meta["quantity"] = rng.randint(1, 5)
        meta["price"] = round(rng.uniform(5.0, 300.0), 2)

    return meta


def generate_events() -> list[dict[str, object]]:
    """Generate all events."""
    rng = random.Random(SEED)
    events: list[dict[str, object]] = []

    for _ in range(NUM_EVENTS):
        user_id = rng.randint(1, NUM_USERS)
        event_type = rng.choice(EVENT_TYPES)
        ts = START_DATE + timedelta(seconds=rng.randint(0, DATE_RANGE_SECONDS))

        event: dict[str, object] = {
            "user_id": user_id,
            "event_type": event_type,
            "timestamp": ts.isoformat(),
            "metadata": generate_metadata(rng, event_type),
        }
        events.append(event)

    return events


def main() -> None:
    print(f"Generating {NUM_EVENTS} events for {NUM_USERS} users...")
    events = generate_events()

    OUTPUT_PATH.write_text(json.dumps(events))
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Written to {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
    sys.exit(0)
