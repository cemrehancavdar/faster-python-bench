"""Cython pipeline: reads raw JSON bytes via yyjson (general-purpose C JSON parser).

Pure Python mode. Uses yyjson for JSON parsing — the C equivalent of Rust's serde.
Both are general-purpose, schema-agnostic parsers. The pipeline walks the parsed
tree with C pointers, filters and aggregates into C structs, then builds Python
dicts only for the final output.

Build: uv run python cython_benchmark/setup_raw.py build_ext --inplace
"""

from __future__ import annotations

import cython
from cython.cimports.cython_benchmark.yyjson import (  # type: ignore[import-not-found]
    yyjson_doc,
    yyjson_val,
    yyjson_arr_iter,
    YYJSON_READ_NOFLAG,
    yyjson_read,
    yyjson_doc_free,
    yyjson_doc_get_root,
    yyjson_get_str,
    yyjson_get_int,
    yyjson_get_real,
    yyjson_is_null,
    yyjson_arr_iter_init,
    yyjson_arr_iter_has_next,
    yyjson_arr_iter_next,
    yyjson_obj_getn,
)
from cython.cimports.libc.stdlib import malloc, free  # type: ignore[import-not-found]
from cython.cimports.libc.string import memcmp  # type: ignore[import-not-found]

from pathlib import Path

DATA_PATH = str(Path(__file__).parent.parent / "data" / "events.json")


# ---------------------------------------------------------------------------
# Timestamp parsing (pure C arithmetic)
# ---------------------------------------------------------------------------

EPOCH_DAYS: cython.int[60] = cython.declare(cython.int[60])
DAYS_BEFORE_MONTH: cython.int[13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

_acc: cython.int = 0
_i: cython.int
for _i in range(60):
    EPOCH_DAYS[_i] = _acc
    _yr: cython.int = 1970 + _i
    if (_yr % 4 == 0 and _yr % 100 != 0) or _yr % 400 == 0:
        _acc += 366
    else:
        _acc += 365


@cython.cfunc
@cython.inline
@cython.exceptval(-1, check=False)
def _d(s: cython.p_const_char, offset: cython.int) -> cython.int:
    """Parse single digit from C string at offset."""
    return s[offset] - 48


@cython.cfunc
@cython.exceptval(-1.0, check=False)
def parse_timestamp(iso: cython.p_const_char) -> cython.double:
    """Parse ISO 8601 timestamp to epoch seconds. Pure C, no Python."""
    year: cython.int = _d(iso, 0) * 1000 + _d(iso, 1) * 100 + _d(iso, 2) * 10 + _d(iso, 3)
    month: cython.int = _d(iso, 5) * 10 + _d(iso, 6)
    day: cython.int = _d(iso, 8) * 10 + _d(iso, 9)
    hour: cython.int = _d(iso, 11) * 10 + _d(iso, 12)
    minute: cython.int = _d(iso, 14) * 10 + _d(iso, 15)
    second: cython.int = _d(iso, 17) * 10 + _d(iso, 18)

    days: cython.int = EPOCH_DAYS[year - 1970] + DAYS_BEFORE_MONTH[month] + day - 1
    if month > 2:
        if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
            days += 1

    return cython.cast(cython.double, days * 86400 + hour * 3600 + minute * 60 + second)


# ---------------------------------------------------------------------------
# Event type matching
# ---------------------------------------------------------------------------

ET_NONE: cython.int = -1
ET_PAGE_VIEW: cython.int = 0
ET_PURCHASE: cython.int = 1
ET_CLICK: cython.int = 2
ET_SEARCH: cython.int = 3
ET_ADD_TO_CART: cython.int = 4
NUM_ET: cython.int = 5

ET_NAMES: tuple[str, ...] = ("PAGE_VIEW", "PURCHASE", "CLICK", "SEARCH", "ADD_TO_CART")


@cython.cfunc
@cython.exceptval(-2, check=False)
def match_event_type(et: cython.p_const_char) -> cython.int:
    """Match event_type C string. Returns ET_ constant or ET_NONE."""
    c0: cython.char = et[0]
    if c0 == ord("p"):
        if memcmp(et, b"page_view", 9) == 0:
            return ET_PAGE_VIEW
        if memcmp(et, b"purchase", 8) == 0:
            return ET_PURCHASE
    elif c0 == ord("c"):
        if memcmp(et, b"click", 5) == 0:
            return ET_CLICK
    elif c0 == ord("s"):
        if memcmp(et, b"search", 6) == 0:
            return ET_SEARCH
    elif c0 == ord("a"):
        if memcmp(et, b"add_to_cart", 11) == 0:
            return ET_ADD_TO_CART
    return ET_NONE


# ---------------------------------------------------------------------------
# Page interning
# ---------------------------------------------------------------------------

NUM_PAGES: cython.int = 11

PAGE_NAMES: tuple[str, ...] = (
    "/about",
    "/account",
    "/blog",
    "/cart",
    "/checkout",
    "/contact",
    "/home",
    "/products",
    "/products/123",
    "/search",
    "",
)


@cython.cfunc
@cython.exceptval(-2, check=False)
def match_page(s: cython.p_const_char) -> cython.int:
    """Match page C string to index into PAGE_NAMES."""
    if s[0] != ord("/"):
        return 10
    c1: cython.char = s[1]
    if c1 == ord("a"):
        if memcmp(s, b"/about", 6) == 0:
            return 0
        if memcmp(s, b"/account", 8) == 0:
            return 1
    elif c1 == ord("b"):
        return 2
    elif c1 == ord("c"):
        if s[2] == ord("a"):
            return 3
        if s[2] == ord("h"):
            return 4
        return 5
    elif c1 == ord("h"):
        return 6
    elif c1 == ord("p"):
        if s[10] == ord("/"):
            return 8
        return 7
    elif c1 == ord("s"):
        return 9
    return 10


# ---------------------------------------------------------------------------
# Per-user aggregation (C struct)
# ---------------------------------------------------------------------------

UserAgg = cython.struct(
    user_id=cython.int,
    event_count=cython.int,
    total_amount=cython.double,
    high_value_count=cython.int,
    first_seen=cython.double,
    last_seen=cython.double,
    et_counts=cython.int[5],
    page_counts=cython.int[11],
)


@cython.cfunc
def user_init(u: cython.pointer(UserAgg), uid: cython.int, ts: cython.double) -> cython.void:
    u.user_id = uid
    u.event_count = 0
    u.total_amount = 0.0
    u.high_value_count = 0
    u.first_seen = ts
    u.last_seen = ts
    i: cython.int
    for i in range(5):
        u.et_counts[i] = 0
    for i in range(11):
        u.page_counts[i] = 0


@cython.cfunc
def user_add(
    u: cython.pointer(UserAgg),
    ts: cython.double,
    amount: cython.double,
    et_idx: cython.int,
    page_idx: cython.int,
) -> cython.void:
    u.event_count += 1
    u.total_amount += amount
    if amount > 100.0:
        u.high_value_count += 1
    if ts < u.first_seen:
        u.first_seen = ts
    if ts > u.last_seen:
        u.last_seen = ts
    u.et_counts[et_idx] += 1
    u.page_counts[page_idx] += 1


# ---------------------------------------------------------------------------
# Hash map: user_id -> UserAgg (open addressing)
# ---------------------------------------------------------------------------

HASH_MASK: cython.uint = 8191

UserSlot = cython.struct(
    user_id=cython.int,
    agg=UserAgg,
)

UserMap = cython.struct(
    slots=UserSlot[8192],
    count=cython.int,
)


@cython.cfunc
def umap_init(m: cython.pointer(UserMap)) -> cython.void:
    m.count = 0
    i: cython.int
    for i in range(8192):
        m.slots[i].user_id = 0


@cython.cfunc
def umap_get(m: cython.pointer(UserMap), uid: cython.int, ts: cython.double) -> cython.pointer(
    UserAgg
):
    h: cython.uint = cython.cast(cython.uint, uid) & HASH_MASK
    while True:
        if m.slots[h].user_id == 0:
            m.slots[h].user_id = uid
            user_init(cython.address(m.slots[h].agg), uid, ts)
            m.count += 1
            return cython.address(m.slots[h].agg)
        elif m.slots[h].user_id == uid:
            return cython.address(m.slots[h].agg)
        h = (h + 1) & HASH_MASK


# ---------------------------------------------------------------------------
# Filter constants
# ---------------------------------------------------------------------------

FILTER_START_TS: cython.double = 1740787200.0
FILTER_END_TS: cython.double = 1759276799.0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_and_run(path: str = DATA_PATH) -> dict:
    """Load raw JSON bytes and run full pipeline. Returns Python dict."""
    raw_bytes: bytes
    with open(path, "rb") as f:
        raw_bytes = f.read()
    return run_pipeline_raw(raw_bytes)


@cython.ccall
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def run_pipeline_raw(json_bytes: bytes) -> dict:
    """Parse raw JSON bytes with yyjson, filter+aggregate in C."""
    data: cython.p_const_char = json_bytes
    data_len: cython.size_t = cython.cast(cython.size_t, len(json_bytes))

    # Parse entire JSON with yyjson (general-purpose C JSON parser)
    doc: cython.pointer(yyjson_doc) = yyjson_read(data, data_len, YYJSON_READ_NOFLAG)
    if doc is cython.NULL:
        raise ValueError("Failed to parse JSON")

    root: cython.pointer(yyjson_val) = yyjson_doc_get_root(doc)
    arr_iter: yyjson_arr_iter = cython.declare(yyjson_arr_iter)
    yyjson_arr_iter_init(root, cython.address(arr_iter))

    # Allocate user map on heap
    um: cython.pointer(UserMap) = cython.cast(
        cython.pointer(UserMap), malloc(cython.sizeof(UserMap))
    )
    if um is cython.NULL:
        yyjson_doc_free(doc)
        raise MemoryError("Failed to allocate user map")
    umap_init(um)

    event_val: cython.pointer(yyjson_val)
    et_val: cython.pointer(yyjson_val)
    ts_val: cython.pointer(yyjson_val)
    uid_val: cython.pointer(yyjson_val)
    meta_val: cython.pointer(yyjson_val)
    amount_val: cython.pointer(yyjson_val)
    page_val: cython.pointer(yyjson_val)
    et_str: cython.p_const_char
    ts_str: cython.p_const_char
    page_str: cython.p_const_char
    uid: cython.int
    ts: cython.double
    amount: cython.double
    et_idx: cython.int
    page_idx: cython.int
    agg: cython.pointer(UserAgg)

    while yyjson_arr_iter_has_next(cython.address(arr_iter)):
        event_val = yyjson_arr_iter_next(cython.address(arr_iter))

        # Get event_type
        et_val = yyjson_obj_getn(event_val, b"event_type", 10)
        et_str = yyjson_get_str(et_val)
        et_idx = match_event_type(et_str)

        if et_idx == ET_NONE:
            continue

        # Parse timestamp
        ts_val = yyjson_obj_getn(event_val, b"timestamp", 9)
        ts_str = yyjson_get_str(ts_val)
        ts = parse_timestamp(ts_str)

        if ts < FILTER_START_TS or ts > FILTER_END_TS:
            continue

        # User ID
        uid_val = yyjson_obj_getn(event_val, b"user_id", 7)
        uid = yyjson_get_int(uid_val)

        # Metadata
        meta_val = yyjson_obj_getn(event_val, b"metadata", 8)

        # Amount
        amount_val = yyjson_obj_getn(meta_val, b"amount", 6)
        if amount_val is not cython.NULL and not yyjson_is_null(amount_val):
            amount = yyjson_get_real(amount_val)
        else:
            amount = 0.0

        # Page
        page_val = yyjson_obj_getn(meta_val, b"page", 4)
        if page_val is not cython.NULL and not yyjson_is_null(page_val):
            page_str = yyjson_get_str(page_val)
            page_idx = match_page(page_str)
        else:
            page_idx = 10

        # Aggregate
        agg = umap_get(um, uid, ts)
        user_add(agg, ts, amount, et_idx, page_idx)

    # --- Build Python result ---
    result_users: list = []
    total_events: cython.int = 0
    total_amount_sum: cython.double = 0.0
    total_high_value: cython.int = 0

    user_ids: list = []
    i: cython.int
    for i in range(8192):
        if um.slots[i].user_id != 0:
            user_ids.append(um.slots[i].user_id)
    user_ids.sort()

    best_et: cython.int
    best_et_count: cython.int
    best_pg: cython.int
    best_pg_count: cython.int
    j: cython.int
    h: cython.uint

    for uid_py in user_ids:
        uid = cython.cast(cython.int, uid_py)
        h = cython.cast(cython.uint, uid) & HASH_MASK
        while um.slots[h].user_id != uid:
            h = (h + 1) & HASH_MASK
        agg = cython.address(um.slots[h].agg)

        best_et = 0
        best_et_count = 0
        for j in range(5):
            if agg.et_counts[j] > best_et_count:
                best_et_count = agg.et_counts[j]
                best_et = j

        best_pg = 0
        best_pg_count = 0
        for j in range(11):
            if agg.page_counts[j] > best_pg_count:
                best_pg_count = agg.page_counts[j]
                best_pg = j

        result_users.append(
            {
                "user_id": agg.user_id,
                "event_count": agg.event_count,
                "total_amount": round(agg.total_amount, 2),
                "high_value_count": agg.high_value_count,
                "duration_seconds": round(agg.last_seen - agg.first_seen, 2),
                "top_event_type": ET_NAMES[best_et],
                "top_page": PAGE_NAMES[best_pg],
            }
        )

        total_events += agg.event_count
        total_amount_sum += agg.total_amount
        total_high_value += agg.high_value_count

    free(um)
    yyjson_doc_free(doc)

    return {
        "total_users": len(result_users),
        "total_events": total_events,
        "total_amount": round(total_amount_sum, 2),
        "total_high_value": total_high_value,
        "users": result_users,
    }
