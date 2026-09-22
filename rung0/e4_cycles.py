"""E4 - hold-and-wait cycles (H4). Rebuilds wait-for graphs from `wait` spans alone
(attrs.holds / attrs.waits_for) and detects cycles independently of the simulator's own
detector, so the same script is the deadlock test-oracle on real traces.

A cycle needs: waiter A holds X and waits for Y, while some B holds Y and waits for X (or a
longer chain). Reported: waits, cycles, sessions that died of deadlock, and the resource
pairs involved.

  python rung0/e4_cycles.py data/synthetic/hosted/*/traces.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter

from observer import load, roots


def cycles_at(waits_open: dict[str, tuple[str, set[str]]]) -> list[tuple[str, ...]]:
    """waits_open: sid -> (waits_for, holds). Edge A->B if B holds what A waits for and B is
    itself waiting. Returns one cycle per start node that lies on a cycle (BFS with parents)."""
    holders: dict[str, set[str]] = {}
    for sid, (_, holds) in waits_open.items():
        for r in holds:
            holders.setdefault(r, set()).add(sid)
    found, seen_keys = [], set()
    for start in waits_open:
        parent = {start: None}
        queue = [start]
        hit = None
        while queue and hit is None:
            node = queue.pop(0)
            for nxt in holders.get(waits_open[node][0], ()):
                if nxt == start:
                    hit = node
                    break
                if nxt in waits_open and nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        if hit is None:
            continue
        path, cur = [], hit
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        key = tuple(sorted(path))
        if key not in seen_keys:
            seen_keys.add(key)
            found.append(tuple(reversed(path)))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    args = ap.parse_args()
    spans = load(args.traces)
    waits = sorted((s for s in spans if s.op == "wait"), key=lambda s: s.t_start)
    rt = roots(spans)
    events = []                                                     # (t, +1/-1, sid, waits_for, holds)
    for w in waits:
        events.append((w.t_start, 1, w.trace_id, w.attrs["waits_for"], set(w.attrs["holds"])))
        events.append((w.t_end, -1, w.trace_id, w.attrs["waits_for"], set(w.attrs["holds"])))
    events.sort(key=lambda e: (e[0], e[1]))
    open_waits: dict[str, tuple[str, set[str]]] = {}
    cycles, pair_counter = [], Counter()
    for t, kind, sid, wf, holds in events:
        if kind == -1:
            open_waits.pop(sid, None)
            continue
        open_waits[sid] = (wf, holds)
        for cyc in cycles_at(open_waits):
            if sid in cyc:
                cycles.append((t, cyc))
                pair_counter[tuple(sorted({open_waits[s][0] for s in cyc}))] += 1
    breaks = sum(1 for w in waits if w.outcome == "deadlock")
    distinct = len({frozenset(c) for _, c in cycles})
    by_res = Counter(w.attrs["waits_for"] for w in waits)
    print(f"wait spans={len(waits)}  by resource={dict(by_res)}")
    print(f"cycle events (independent reconstruction)={len(cycles)}  distinct session-sets in cycles={distinct}  "
          f"waits broken as deadlock={breaks} (client timeouts usually fire first)")
    if pair_counter:
        print("resources in cycles:", dict(pair_counter))
    flagged = sum(1 for w in waits if w.attrs.get("deadlock_cycle"))
    print(f"simulator-flagged cycle waits={flagged}")
    print("\nH4 verdict:", "CONFIRMED - hold-and-wait across tiers forms real cycles; the controller must break them by ordering + all-or-nothing + expiry"
          if cycles else "no cycles on this data - keep global ordering (free) and drop gang acquisition (v1 §8 E4)")


if __name__ == "__main__":
    main()
