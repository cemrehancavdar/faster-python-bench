"""PyPy rung: same as baseline, run under PyPy interpreter.

This is identical to baseline/pipeline.py. The point is that PyPy's
JIT accelerates the same pure Python code with zero changes.

Run: pypy3 pypy_benchmark/pipeline.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent.parent / "data" / "events.json"

FILTER_START = "2025-03-01T00:00:00+00:00"
FILTER_END = "2025-09-30T23:59:59+00:00"
FILTER_EVENT_TYPES = {"page_view", "purchase", "click", "search", "add_to_cart"}

_FILTER_START_TS = datetime.fromisoformat(FILTER_START).timestamp()
_FILTER_END_TS = datetime.fromisoformat(FILTER_END).timestamp()


def load_events(path: str | Path = DATA_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def parse_iso_timestamp(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def run_pipeline(events: list[dict[str, Any]]) -> dict[str, Any]:
    # --- Filter ---
    filtered: list[dict[str, Any]] = []
    for event in events:
        if event["event_type"] not in FILTER_EVENT_TYPES:
            continue
        ts = parse_iso_timestamp(event["timestamp"])
        if ts < _FILTER_START_TS or ts > _FILTER_END_TS:
            continue
        filtered.append(event)

    # --- Transform ---
    transformed: list[dict[str, Any]] = []
    for event in filtered:
        meta = event["metadata"]
        amount = float(meta.get("amount", 0.0)) if "amount" in meta else 0.0
        ts = parse_iso_timestamp(event["timestamp"])
        record = {
            "user_id": event["user_id"],
            "event_type": event["event_type"].upper(),
            "timestamp_epoch": ts,
            "page": meta.get("page", ""),
            "amount": amount,
            "is_high_value": amount > 100.0,
        }
        transformed.append(record)

    # --- Aggregate ---
    users: dict[int, dict[str, Any]] = {}
    for rec in transformed:
        uid = rec["user_id"]
        if uid not in users:
            users[uid] = {
                "user_id": uid,
                "event_count": 0,
                "total_amount": 0.0,
                "high_value_count": 0,
                "first_seen": rec["timestamp_epoch"],
                "last_seen": rec["timestamp_epoch"],
                "event_types": Counter(),
                "pages": Counter(),
            }
        u = users[uid]
        u["event_count"] += 1
        u["total_amount"] += rec["amount"]
        if rec["is_high_value"]:
            u["high_value_count"] += 1
        if rec["timestamp_epoch"] < u["first_seen"]:
            u["first_seen"] = rec["timestamp_epoch"]
        if rec["timestamp_epoch"] > u["last_seen"]:
            u["last_seen"] = rec["timestamp_epoch"]
        u["event_types"][rec["event_type"]] += 1
        u["pages"][rec["page"]] += 1

    # --- Finalize ---
    result_users: list[dict[str, Any]] = []
    total_events = 0
    total_amount = 0.0
    total_high_value = 0

    for u in users.values():
        duration = u["last_seen"] - u["first_seen"]
        top_event = u["event_types"].most_common(1)[0][0] if u["event_types"] else ""
        top_page = u["pages"].most_common(1)[0][0] if u["pages"] else ""
        result_users.append(
            {
                "user_id": u["user_id"],
                "event_count": u["event_count"],
                "total_amount": round(u["total_amount"], 2),
                "high_value_count": u["high_value_count"],
                "duration_seconds": round(duration, 2),
                "top_event_type": top_event,
                "top_page": top_page,
            }
        )
        total_events += u["event_count"]
        total_amount += u["total_amount"]
        total_high_value += u["high_value_count"]

    result_users.sort(key=lambda x: x["user_id"])

    return {
        "total_users": len(result_users),
        "total_events": total_events,
        "total_amount": round(total_amount, 2),
        "total_high_value": total_high_value,
        "users": result_users,
    }


def main() -> None:
    events = load_events()
    result = run_pipeline(events)
    print(f"Users:       {result['total_users']}")
    print(f"Events:      {result['total_events']}")
    print(f"Amount:      ${result['total_amount']:,.2f}")
    print(f"High-value:  {result['total_high_value']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
