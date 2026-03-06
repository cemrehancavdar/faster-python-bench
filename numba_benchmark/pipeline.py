"""Numba rung: JIT-compiled pipeline with pre-encoded arrays.

Key optimizations vs baseline:
- Pre-encode ALL data to numpy arrays before JIT:
  - Timestamps: pre-parsed to float64 epoch seconds
  - Event types: encoded as int8 codes
  - Pages: encoded as int8 codes
  - Amounts: float64 array
  - User IDs: int64 array
- The @njit function does filter + aggregate in one pass over arrays
- String operations (upper, set membership) handled during encoding
- The JIT function is pure numeric — no Python objects, no dicts, no strings

Run: uv run --extra numba numba_benchmark/pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numba import (
    njit,  # type: ignore[import-untyped]
)

DATA_PATH = Path(__file__).parent.parent / "data" / "events.json"

FILTER_EVENT_TYPES = {"page_view", "purchase", "click", "search", "add_to_cart"}

FILTER_START_TS = 1740787200.0  # 2025-03-01T00:00:00 UTC
FILTER_END_TS = 1759276799.0  # 2025-09-30T23:59:59 UTC
HIGH_VALUE_THRESHOLD = 100.0

# Epoch days lookup
_EPOCH_DAYS_LIST: list[int] = []
_d = 0
for _y in range(1970, 2030):
    _EPOCH_DAYS_LIST.append(_d)
    if (_y % 4 == 0 and _y % 100 != 0) or _y % 400 == 0:
        _d += 366
    else:
        _d += 365
EPOCH_DAYS = np.array(_EPOCH_DAYS_LIST, dtype=np.int64)
DAYS_BEFORE_MONTH = np.array(
    [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=np.int64
)


def load_events(path: str | Path = DATA_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def parse_timestamp_py(iso: str) -> float:
    """Parse ISO timestamp — Python version for encoding phase."""
    y = (
        (ord(iso[0]) - 48) * 1000
        + (ord(iso[1]) - 48) * 100
        + (ord(iso[2]) - 48) * 10
        + (ord(iso[3]) - 48)
    )
    mo = (ord(iso[5]) - 48) * 10 + (ord(iso[6]) - 48)
    d = (ord(iso[8]) - 48) * 10 + (ord(iso[9]) - 48)
    h = (ord(iso[11]) - 48) * 10 + (ord(iso[12]) - 48)
    mi = (ord(iso[14]) - 48) * 10 + (ord(iso[15]) - 48)
    s = (ord(iso[17]) - 48) * 10 + (ord(iso[18]) - 48)
    days = (
        _EPOCH_DAYS_LIST[y - 1970]
        + [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][mo]
        + d
        - 1
    )
    if mo > 2 and ((y % 4 == 0 and y % 100 != 0) or y % 400 == 0):
        days += 1
    return float(days * 86400 + h * 3600 + mi * 60 + s)


def encode_events(
    events: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Encode all events to parallel numpy arrays.

    Returns:
        user_ids (int64), timestamps (float64), amounts (float64),
        event_type_codes (int8), page_codes (int8),
        event_type_names (list), page_names (list)
    """
    # Build string -> int code maps
    event_type_to_code: dict[str, int] = {}
    page_to_code: dict[str, int] = {}
    filter_codes: set[int] = set()

    # Pre-scan to build code tables
    for et in FILTER_EVENT_TYPES:
        code = len(event_type_to_code)
        event_type_to_code[et] = code
        filter_codes.add(code)

    n = len(events)
    user_ids = np.empty(n, dtype=np.int64)
    timestamps = np.empty(n, dtype=np.float64)
    amounts = np.empty(n, dtype=np.float64)
    event_codes = np.empty(n, dtype=np.int8)
    page_codes = np.empty(n, dtype=np.int16)

    for i, event in enumerate(events):
        user_ids[i] = event["user_id"]
        timestamps[i] = parse_timestamp_py(event["timestamp"])

        et = event["event_type"]
        if et not in event_type_to_code:
            event_type_to_code[et] = len(event_type_to_code)
        event_codes[i] = event_type_to_code[et]

        meta = event["metadata"]
        amounts[i] = float(meta["amount"]) if "amount" in meta else 0.0

        page = meta.get("page", "")
        if page not in page_to_code:
            page_to_code[page] = len(page_to_code)
        page_codes[i] = page_to_code[page]

    # Reverse maps for output
    et_names = [""] * len(event_type_to_code)
    for name, code in event_type_to_code.items():
        et_names[code] = name.upper()

    pg_names = [""] * len(page_to_code)
    for name, code in page_to_code.items():
        pg_names[code] = name

    return user_ids, timestamps, amounts, event_codes, page_codes, et_names, pg_names


@njit(cache=True)  # type: ignore[misc]
def pipeline_jit(
    user_ids: np.ndarray,
    timestamps: np.ndarray,
    amounts: np.ndarray,
    event_codes: np.ndarray,
    page_codes: np.ndarray,
    filter_codes: np.ndarray,
    filter_start: float,
    filter_end: float,
    high_value_threshold: float,
    max_uid: int,
    num_event_types: int,
    num_pages: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Pure numeric pipeline: filter + aggregate in one pass.

    Returns per-user arrays indexed by user_id:
    (counts, total_amounts, hv_counts, first_seen, last_seen,
     active flags, event_type_hist, page_hist)
    """
    size = max_uid + 1
    counts = np.zeros(size, dtype=np.int64)
    total_amounts = np.zeros(size, dtype=np.float64)
    hv_counts = np.zeros(size, dtype=np.int64)
    first_seen = np.full(size, 1e18, dtype=np.float64)
    last_seen = np.full(size, -1e18, dtype=np.float64)
    active = np.zeros(size, dtype=np.int8)

    # Per-user histograms: [uid, code] -> count
    et_hist = np.zeros((size, num_event_types), dtype=np.int64)
    pg_hist = np.zeros((size, num_pages), dtype=np.int64)

    n = len(user_ids)
    num_filter = len(filter_codes)

    for i in range(n):
        # Check if event type is in filter set
        ec = event_codes[i]
        is_filter = False
        for fi in range(num_filter):
            if ec == filter_codes[fi]:
                is_filter = True
                break
        if not is_filter:
            continue

        ts = timestamps[i]
        if ts < filter_start or ts > filter_end:
            continue

        uid = user_ids[i]
        amount = amounts[i]

        active[uid] = 1
        counts[uid] += 1
        total_amounts[uid] += amount
        if amount > high_value_threshold:
            hv_counts[uid] += 1
        if ts < first_seen[uid]:
            first_seen[uid] = ts
        if ts > last_seen[uid]:
            last_seen[uid] = ts

        et_hist[uid, ec] += 1
        pg_hist[uid, page_codes[i]] += 1

    return counts, total_amounts, hv_counts, first_seen, last_seen, active, et_hist, pg_hist


def run_pipeline(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Full pipeline: encode -> JIT filter+aggregate -> finalize."""
    user_ids, timestamps, amounts, event_codes, page_codes, et_names, pg_names = encode_events(
        events
    )

    max_uid = int(user_ids.max())
    num_et = len(et_names)
    num_pg = len(pg_names)

    # Build filter codes array
    filter_et_set = {et.upper() for et in FILTER_EVENT_TYPES}
    filter_codes_list = [i for i, name in enumerate(et_names) if name in filter_et_set]
    filter_codes = np.array(filter_codes_list, dtype=np.int8)

    counts, total_amounts, hv_counts, first_seen, last_seen, active, et_hist, pg_hist = (
        pipeline_jit(
            user_ids,
            timestamps,
            amounts,
            event_codes,
            page_codes,
            filter_codes,
            FILTER_START_TS,
            FILTER_END_TS,
            HIGH_VALUE_THRESHOLD,
            max_uid,
            num_et,
            num_pg,
        )
    )

    # Finalize (Python — build output dicts)
    result_users: list[dict[str, Any]] = []
    total_events = 0
    total_amount = 0.0
    total_high_value = 0

    for uid in range(max_uid + 1):
        if not active[uid]:
            continue

        # Find most common event type and page
        top_et_code = int(np.argmax(et_hist[uid]))
        top_pg_code = int(np.argmax(pg_hist[uid]))
        top_event = et_names[top_et_code] if et_hist[uid, top_et_code] > 0 else ""
        top_page = pg_names[top_pg_code] if pg_hist[uid, top_pg_code] > 0 else ""

        duration = float(last_seen[uid] - first_seen[uid])

        result_users.append(
            {
                "user_id": uid,
                "event_count": int(counts[uid]),
                "total_amount": round(float(total_amounts[uid]), 2),
                "high_value_count": int(hv_counts[uid]),
                "duration_seconds": round(duration, 2),
                "top_event_type": top_event,
                "top_page": top_page,
            }
        )

        total_events += int(counts[uid])
        total_amount += float(total_amounts[uid])
        total_high_value += int(hv_counts[uid])

    result_users.sort(key=lambda x: x["user_id"])

    return {
        "total_users": len(result_users),
        "total_events": total_events,
        "total_amount": round(total_amount, 2),
        "total_high_value": total_high_value,
        "users": result_users,
    }


def warmup(events: list[dict[str, Any]]) -> None:
    """Trigger Numba JIT compilation."""
    run_pipeline(events[:200])


def main() -> None:
    events = load_events()
    warmup(events)
    result = run_pipeline(events)
    print(f"Users:       {result['total_users']}")
    print(f"Events:      {result['total_events']}")
    print(f"Amount:      ${result['total_amount']:,.2f}")
    print(f"High-value:  {result['total_high_value']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
