"""Mypyc rung: type-annotated Python compiled to C via mypyc.

Key optimizations vs baseline:
- Manual timestamp parsing with pure integer arithmetic (no datetime)
- Typed dataclass-like UserAgg class with typed attributes (mypyc compiles
  attribute access to direct C struct field access — much faster than dict)
- Pre-computed lookup tables as module-level tuples
- Explicit int/float types on all locals so mypyc can use C primitives
- Avoid 'Any' — mypyc can't optimize dynamic types

Build: uv run python -c "from mypyc.build import mypycify; ..." build_ext --inplace
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    _THIS_DIR = Path(__file__).parent
except (KeyError, TypeError):
    _THIS_DIR = Path.cwd() / "mypyc_benchmark"
DATA_PATH = _THIS_DIR.parent / "data" / "events.json"

FILTER_EVENT_TYPES: frozenset[str] = frozenset(
    ["page_view", "purchase", "click", "search", "add_to_cart"]
)

# Pre-computed epoch days to year start (tuple for mypyc — faster than dict)
# Index: year - 1970
_EPOCH_DAYS_LIST: list[int] = []
_d = 0
for _y in range(1970, 2030):
    _EPOCH_DAYS_LIST.append(_d)
    if (_y % 4 == 0 and _y % 100 != 0) or _y % 400 == 0:
        _d += 366
    else:
        _d += 365
EPOCH_DAYS: tuple[int, ...] = tuple(_EPOCH_DAYS_LIST)

DAYS_BEFORE_MONTH: tuple[int, ...] = (0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)

FILTER_START_TS: float = 1740787200.0
FILTER_END_TS: float = 1759276799.0
HIGH_VALUE_THRESHOLD: float = 100.0


def load_events(path: str = "") -> list[dict[str, object]]:
    if not path:
        path = str(DATA_PATH)
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def parse_timestamp(iso: str) -> float:
    """Parse ISO 8601 to epoch seconds using integer arithmetic.

    mypyc compiles this to C integer operations — much faster than
    calling datetime.fromisoformat() which has Python-to-C bridge overhead.
    """
    # ord() - 48 converts digit char to int at C speed under mypyc
    year: int = (
        (ord(iso[0]) - 48) * 1000
        + (ord(iso[1]) - 48) * 100
        + (ord(iso[2]) - 48) * 10
        + (ord(iso[3]) - 48)
    )
    month: int = (ord(iso[5]) - 48) * 10 + (ord(iso[6]) - 48)
    day: int = (ord(iso[8]) - 48) * 10 + (ord(iso[9]) - 48)
    hour: int = (ord(iso[11]) - 48) * 10 + (ord(iso[12]) - 48)
    minute: int = (ord(iso[14]) - 48) * 10 + (ord(iso[15]) - 48)
    second: int = (ord(iso[17]) - 48) * 10 + (ord(iso[18]) - 48)

    days: int = EPOCH_DAYS[year - 1970] + DAYS_BEFORE_MONTH[month] + day - 1
    if month > 2 and ((year % 4 == 0 and year % 100 != 0) or year % 400 == 0):
        days += 1

    return float(days * 86400 + hour * 3600 + minute * 60 + second)


class UserAgg:
    """Per-user aggregation with typed attributes.

    mypyc compiles attribute access to direct C struct field access.
    This is MUCH faster than dict["key"] lookups.
    """

    def __init__(self, user_id: int, first_ts: float) -> None:
        self.user_id: int = user_id
        self.event_count: int = 0
        self.total_amount: float = 0.0
        self.high_value_count: int = 0
        self.first_seen: float = first_ts
        self.last_seen: float = first_ts
        # String counters — mypyc handles dict[str, int] well
        self.event_type_counts: dict[str, int] = {}
        self.page_counts: dict[str, int] = {}

    def add_event(self, ts: float, amount: float, is_hv: bool, event_type: str, page: str) -> None:
        self.event_count += 1
        self.total_amount += amount
        if is_hv:
            self.high_value_count += 1
        if ts < self.first_seen:
            self.first_seen = ts
        if ts > self.last_seen:
            self.last_seen = ts

        # Inline counter update (avoids Counter overhead)
        etc = self.event_type_counts
        etc[event_type] = etc.get(event_type, 0) + 1
        pc = self.page_counts
        pc[page] = pc.get(page, 0) + 1


def most_common_key(d: dict[str, int]) -> str:
    """Find the key with the highest value."""
    best_key: str = ""
    best_val: int = 0
    for k, v in d.items():
        if v > best_val:
            best_val = v
            best_key = k
    return best_key


def run_pipeline(events: list[dict[str, object]]) -> dict[str, object]:
    """Full pipeline optimized for mypyc."""
    users: dict[int, UserAgg] = {}

    # --- Single pass: filter + transform + aggregate ---
    for event in events:
        event_type_obj = event["event_type"]
        assert isinstance(event_type_obj, str)
        event_type: str = event_type_obj

        if event_type not in FILTER_EVENT_TYPES:
            continue

        ts_obj = event["timestamp"]
        assert isinstance(ts_obj, str)
        ts: float = parse_timestamp(ts_obj)

        if ts < FILTER_START_TS or ts > FILTER_END_TS:
            continue

        meta_obj = event["metadata"]
        assert isinstance(meta_obj, dict)
        meta: dict[str, object] = meta_obj

        amount: float = 0.0
        if "amount" in meta:
            amount_obj = meta["amount"]
            if isinstance(amount_obj, float):
                amount = amount_obj
            elif isinstance(amount_obj, int):
                amount = float(amount_obj)

        uid_obj = event["user_id"]
        assert isinstance(uid_obj, int)
        uid: int = uid_obj

        is_hv: bool = amount > HIGH_VALUE_THRESHOLD
        et_upper: str = event_type.upper()

        page_obj = meta.get("page", "")
        page: str = str(page_obj) if page_obj is not None else ""

        if uid not in users:
            users[uid] = UserAgg(uid, ts)

        users[uid].add_event(ts, amount, is_hv, et_upper, page)

    # --- Finalize ---
    result_users: list[dict[str, object]] = []
    total_events: int = 0
    total_amount: float = 0.0
    total_high_value: int = 0

    sorted_uids: list[int] = sorted(users.keys())
    for uid in sorted_uids:
        u: UserAgg = users[uid]
        duration: float = u.last_seen - u.first_seen
        top_event: str = most_common_key(u.event_type_counts)
        top_page: str = most_common_key(u.page_counts)

        result_users.append(
            {
                "user_id": u.user_id,
                "event_count": u.event_count,
                "total_amount": round(u.total_amount, 2),
                "high_value_count": u.high_value_count,
                "duration_seconds": round(duration, 2),
                "top_event_type": top_event,
                "top_page": top_page,
            }
        )

        total_events += u.event_count
        total_amount += u.total_amount
        total_high_value += u.high_value_count

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
