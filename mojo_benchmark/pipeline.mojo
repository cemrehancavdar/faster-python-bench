"""Mojo rung: Python-inspired systems language.

This implements the same JSON transform pipeline in Mojo.
Mojo looks like Python but is a different language with different
semantics (value types, ownership, SIMD, etc.).

Run: mojo run mojo_bench/pipeline.mojo

Note: Mojo's JSON support and stdlib are still evolving. This
implementation uses Mojo's built-in types and manual JSON parsing
where needed. The code is intentionally kept simple and close to
the Python version for comparison.

As of early 2026, Mojo's ecosystem is young. This file may need
updates as the language evolves.
"""

from collections import Dict, List
from pathlib import Path
from time import perf_counter
import sys


# Mojo doesn't have a mature JSON library yet. For a fair benchmark,
# we'd need to either:
# 1. Use Mojo's Python interop to call json.load() (defeats the purpose)
# 2. Write a JSON parser in Mojo (unfair extra work)
# 3. Pre-convert the data to a Mojo-friendly format
#
# For the blog post, we'll show option 1 (Python interop) to demonstrate
# that Mojo's strength is in the compute loop, not in parsing.
# The benchmark runner will handle this specially.

from python import Python


fn main() raises:
    var json_mod = Python.import_module("json")
    var pathlib = Python.import_module("pathlib")

    var data_path = pathlib.Path(__file__).parent.parent / "data" / "events.json"
    var f = Python.import_module("builtins").open(str(data_path))
    var events = json_mod.load(f)
    f.close()

    var num_events = len(events)
    print("Loaded", num_events, "events")

    # Filter constants
    var filter_start_ts: Float64 = 1740787200.0  # 2025-03-01
    var filter_end_ts: Float64 = 1759276799.0    # 2025-09-30
    var filter_types = Python.import_module("builtins").set(
        ["page_view", "purchase", "click", "search", "add_to_cart"]
    )

    var datetime_mod = Python.import_module("datetime")
    var high_value_threshold: Float64 = 100.0

    # --- Filter + Transform + Aggregate ---
    # Using Python interop for JSON access, Mojo for the logic
    var user_counts = Dict[Int, Int]()
    var user_amounts = Dict[Int, Float64]()
    var user_hv = Dict[Int, Int]()
    var user_first = Dict[Int, Float64]()
    var user_last = Dict[Int, Float64]()

    var start = perf_counter()

    for i in range(num_events):
        var event = events[i]
        var event_type = str(event["event_type"])

        if event_type not in filter_types:
            continue

        var ts_str = str(event["timestamp"])
        var ts = float(datetime_mod.datetime.fromisoformat(ts_str).timestamp())

        if ts < filter_start_ts or ts > filter_end_ts:
            continue

        var uid = int(event["user_id"])
        var meta = event["metadata"]
        var amount: Float64 = 0.0
        if "amount" in meta:
            amount = float(meta["amount"])

        if uid not in user_counts:
            user_counts[uid] = 0
            user_amounts[uid] = 0.0
            user_hv[uid] = 0
            user_first[uid] = ts
            user_last[uid] = ts

        user_counts[uid] = user_counts[uid] + 1
        user_amounts[uid] = user_amounts[uid] + amount
        if amount > high_value_threshold:
            user_hv[uid] = user_hv[uid] + 1
        if ts < user_first[uid]:
            user_first[uid] = ts
        if ts > user_last[uid]:
            user_last[uid] = ts

    var elapsed = perf_counter() - start

    # Summary
    var total_users = len(user_counts)
    var total_events = 0
    var total_amount: Float64 = 0.0
    var total_hv = 0

    for entry in user_counts.items():
        total_events += entry[].value
    for entry in user_amounts.items():
        total_amount += entry[].value
    for entry in user_hv.items():
        total_hv += entry[].value

    print("Users:      ", total_users)
    print("Events:     ", total_events)
    print("High-value: ", total_hv)
    print("Time:       ", elapsed, "s")
