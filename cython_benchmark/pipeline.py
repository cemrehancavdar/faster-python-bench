"""Cython rung: Pure Python mode — fully optimized for compilation.

Key optimizations vs baseline:
- Manual timestamp parsing using C-typed character arithmetic
- C arrays for lookup tables (not Python tuples)
- Single-pass filter+transform+aggregate (no intermediate list-of-dicts)
- @cython.cclass UserAgg with typed attributes (C struct field access)
- _add_event is @cython.cfunc (pure C call, no Python arg wrapping)
- run_pipeline is @cython.ccall (C-fast internals, Python-callable)
- Module-level constants cached as C locals
- Dict keys pre-interned as local str variables
- Pre-built uppercase map to avoid .upper() in hot loop
- Direct CPython C-API: PyList_GET_ITEM + PyDict_GetItem (borrowed refs)

Build: uv run python cython_benchmark/setup.py build_ext --inplace
"""

from __future__ import annotations

import cython
from cython.cimports.cpython.dict import PyDict_GetItem, PyDict_Contains  # type: ignore[import-not-found]
from cython.cimports.cpython.list import PyList_GET_ITEM, PyList_GET_SIZE  # type: ignore[import-not-found]
from cython.cimports.cpython.object import PyObject  # type: ignore[import-not-found]
from cython.cimports.cpython.float import PyFloat_AS_DOUBLE  # type: ignore[import-not-found]
from cython.cimports.cpython.long import PyLong_AsLong  # type: ignore[import-not-found]
from cython.cimports.cpython.set import PySet_Contains  # type: ignore[import-not-found]
import json
import sys
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent.parent / "data" / "events.json"

FILTER_EVENT_TYPES: set[str] = {"page_view", "purchase", "click", "search", "add_to_cart"}

_UPPER_MAP: dict[str, str] = {
    "page_view": "PAGE_VIEW",
    "purchase": "PURCHASE",
    "click": "CLICK",
    "search": "SEARCH",
    "add_to_cart": "ADD_TO_CART",
}

# C arrays for epoch-day lookup
_EPOCH_DAYS_C: cython.int[60] = cython.declare(cython.int[60])
_i: cython.int
_acc: cython.int = 0
for _i in range(60):
    _EPOCH_DAYS_C[_i] = _acc
    _yr: cython.int = 1970 + _i
    if (_yr % 4 == 0 and _yr % 100 != 0) or _yr % 400 == 0:
        _acc += 366
    else:
        _acc += 365

_DAYS_BEFORE_MONTH_C: cython.int[13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

FILTER_START_TS: cython.double = 1740787200.0
FILTER_END_TS: cython.double = 1759276799.0
HIGH_VALUE_THRESHOLD: cython.double = 100.0


def load_events(path: str | Path = DATA_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


@cython.cfunc
@cython.inline
@cython.exceptval(-1, check=False)
def _parse_digit(c: cython.Py_UCS4) -> cython.int:
    return cython.cast(cython.int, c) - 48


@cython.cfunc
@cython.exceptval(-1.0, check=False)
def _parse_timestamp(iso: str) -> cython.double:
    """Parse ISO 8601 to epoch seconds."""
    c0: cython.Py_UCS4 = iso[0]
    c1: cython.Py_UCS4 = iso[1]
    c2: cython.Py_UCS4 = iso[2]
    c3: cython.Py_UCS4 = iso[3]
    c5: cython.Py_UCS4 = iso[5]
    c6: cython.Py_UCS4 = iso[6]
    c8: cython.Py_UCS4 = iso[8]
    c9: cython.Py_UCS4 = iso[9]
    c11: cython.Py_UCS4 = iso[11]
    c12: cython.Py_UCS4 = iso[12]
    c14: cython.Py_UCS4 = iso[14]
    c15: cython.Py_UCS4 = iso[15]
    c17: cython.Py_UCS4 = iso[17]
    c18: cython.Py_UCS4 = iso[18]

    year: cython.int = (
        _parse_digit(c0) * 1000 + _parse_digit(c1) * 100 + _parse_digit(c2) * 10 + _parse_digit(c3)
    )
    month: cython.int = _parse_digit(c5) * 10 + _parse_digit(c6)
    day: cython.int = _parse_digit(c8) * 10 + _parse_digit(c9)
    hour: cython.int = _parse_digit(c11) * 10 + _parse_digit(c12)
    minute: cython.int = _parse_digit(c14) * 10 + _parse_digit(c15)
    second: cython.int = _parse_digit(c17) * 10 + _parse_digit(c18)

    days: cython.int = _EPOCH_DAYS_C[year - 1970] + _DAYS_BEFORE_MONTH_C[month] + day - 1
    if month > 2:
        if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
            days = days + 1

    return cython.cast(cython.double, days * 86400 + hour * 3600 + minute * 60 + second)


@cython.cclass
class UserAgg:
    """Per-user aggregation with typed attributes (C struct)."""

    user_id: cython.int
    event_count: cython.int
    total_amount: cython.double
    high_value_count: cython.int
    first_seen: cython.double
    last_seen: cython.double
    event_type_counts: dict
    page_counts: dict

    def __init__(self, user_id: cython.int, first_ts: cython.double) -> None:
        self.user_id = user_id
        self.event_count = 0
        self.total_amount = 0.0
        self.high_value_count = 0
        self.first_seen = first_ts
        self.last_seen = first_ts
        self.event_type_counts = {}
        self.page_counts = {}


@cython.cfunc
def _add_event(
    agg: UserAgg,
    ts: cython.double,
    amount: cython.double,
    is_hv: cython.bint,
    event_type: str,
    page: str,
) -> None:
    agg.event_count += 1
    agg.total_amount += amount
    if is_hv:
        agg.high_value_count += 1
    if ts < agg.first_seen:
        agg.first_seen = ts
    if ts > agg.last_seen:
        agg.last_seen = ts
    etc: dict = agg.event_type_counts
    etc[event_type] = etc.get(event_type, 0) + 1
    pc: dict = agg.page_counts
    pc[page] = pc.get(page, 0) + 1


@cython.cfunc
def _most_common_key(d: dict) -> str:
    best_key: str = ""
    best_val: cython.int = 0
    k: str
    v: cython.int
    for k, v in d.items():
        if v > best_val:
            best_val = v
            best_key = k
    return best_key


@cython.ccall
def run_pipeline(events: list) -> dict:
    """Single-pass pipeline using direct CPython C-API for dict/list access."""
    users: dict = {}
    upper_map: dict = _UPPER_MAP

    filter_types: set = FILTER_EVENT_TYPES
    start_ts: cython.double = FILTER_START_TS
    end_ts: cython.double = FILTER_END_TS
    hv_thresh: cython.double = HIGH_VALUE_THRESHOLD

    # Pre-intern key strings (Cython will use their PyObject* directly)
    k_event_type: str = "event_type"
    k_timestamp: str = "timestamp"
    k_user_id: str = "user_id"
    k_metadata: str = "metadata"
    k_amount: str = "amount"
    k_page: str = "page"
    empty: str = ""

    ts: cython.double
    amount: cython.double
    uid: cython.int
    is_hv: cython.bint
    agg: UserAgg
    n: cython.Py_ssize_t = PyList_GET_SIZE(events)
    idx: cython.Py_ssize_t

    # Raw C-API pointers for borrowed references
    p_et: cython.pointer(PyObject)
    p_ts: cython.pointer(PyObject)
    p_uid: cython.pointer(PyObject)
    p_meta: cython.pointer(PyObject)
    p_amount: cython.pointer(PyObject)
    p_page: cython.pointer(PyObject)
    event: dict
    meta: dict
    event_type: str

    for idx in range(n):
        event = cython.cast(dict, cython.cast(object, PyList_GET_ITEM(events, idx)))

        # PyDict_GetItem returns borrowed ref (no incref) — faster than event["key"]
        p_et = PyDict_GetItem(event, k_event_type)
        event_type = cython.cast(str, cython.cast(object, p_et))

        if PySet_Contains(filter_types, event_type) == 0:
            continue

        p_ts = PyDict_GetItem(event, k_timestamp)
        ts = _parse_timestamp(cython.cast(str, cython.cast(object, p_ts)))
        if ts < start_ts or ts > end_ts:
            continue

        p_uid = PyDict_GetItem(event, k_user_id)
        uid = PyLong_AsLong(cython.cast(object, p_uid))

        p_meta = PyDict_GetItem(event, k_metadata)
        meta = cython.cast(dict, cython.cast(object, p_meta))

        p_amount = PyDict_GetItem(meta, k_amount)
        if p_amount is not cython.NULL:
            amount = PyFloat_AS_DOUBLE(cython.cast(object, p_amount))
        else:
            amount = 0.0

        et_upper: str = cython.cast(str, cython.cast(object, PyDict_GetItem(upper_map, event_type)))

        p_page = PyDict_GetItem(meta, k_page)
        if p_page is not cython.NULL:
            page: str = cython.cast(str, cython.cast(object, p_page))
        else:
            page = empty

        is_hv = amount > hv_thresh

        agg = cython.cast(UserAgg, users.get(uid))
        if agg is None:
            agg = UserAgg(uid, ts)
            users[uid] = agg

        _add_event(agg, ts, amount, is_hv, et_upper, page)

    # --- Finalize ---
    result_users: list = []
    total_events: cython.int = 0
    total_amount: cython.double = 0.0
    total_high_value: cython.int = 0

    uid_key: cython.int
    for uid_key in sorted(users.keys()):
        agg = cython.cast(UserAgg, users[uid_key])
        top_event: str = _most_common_key(agg.event_type_counts)
        top_page: str = _most_common_key(agg.page_counts)
        duration: cython.double = agg.last_seen - agg.first_seen

        result_users.append(
            {
                "user_id": agg.user_id,
                "event_count": agg.event_count,
                "total_amount": round(agg.total_amount, 2),
                "high_value_count": agg.high_value_count,
                "duration_seconds": round(duration, 2),
                "top_event_type": top_event,
                "top_page": top_page,
            }
        )

        total_events += agg.event_count
        total_amount += agg.total_amount
        total_high_value += agg.high_value_count

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
