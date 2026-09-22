from pathlib import Path

import pytest

from society.memory import KVStore, LineageLog, LineageRecord, current_key


def test_attempts_are_isolated_and_immutable():
    kv = KVStore()
    kv.put_attempt("r1", "backend", "schema", 1, {"fields": ["token"]})
    kv.put_attempt("r1", "backend", "schema", 2, {"fields": ["token", "userId"]})

    assert kv.get_attempt("r1", "backend", "schema", 1) == {"fields": ["token"]}
    assert kv.get_attempt("r1", "backend", "schema", 2) == {"fields": ["token", "userId"]}
    with pytest.raises(ValueError):
        kv.put_attempt("r1", "backend", "schema", 1, {"overwrite": True})
    assert kv.next_attempt("r1", "backend", "schema") == 3


def test_current_only_moves_on_promote():
    kv = KVStore()
    assert kv.get_current("r1", "backend", "schema") is None

    kv.put_attempt("r1", "backend", "schema", 1, {"v": 1})
    assert kv.get_current("r1", "backend", "schema") is None, "writing an attempt must not move :current"

    kv.promote("r1", "backend", "schema", 1)
    assert kv.get_current("r1", "backend", "schema") == {"v": 1}
    assert kv.current_attempt("r1", "backend", "schema") == 1

    kv.put_attempt("r1", "backend", "schema", 2, {"v": 2})
    assert kv.get_current("r1", "backend", "schema") == {"v": 1}, "rejected retry must not be visible downstream"

    kv.promote("r1", "backend", "schema", 2)
    assert kv.get_current("r1", "backend", "schema") == {"v": 2}


def test_promote_requires_existing_attempt():
    kv = KVStore()
    with pytest.raises(KeyError):
        kv.promote("r1", "backend", "schema", 7)
    assert not kv.exists(current_key("r1", "backend", "schema"))


def test_runs_do_not_leak():
    kv = KVStore()
    kv.put_attempt("r1", "backend", "schema", 1, {"run": 1})
    kv.promote("r1", "backend", "schema", 1)
    assert kv.get_current("r2", "backend", "schema") is None
    assert kv.next_attempt("r2", "backend", "schema") == 1


def test_lineage_is_append_only_and_ordered(tmp_path: Path):
    log = LineageLog(tmp_path / "lineage.jsonl")
    for i, verdict in enumerate(["rejected", "rejected", "approved"], start=1):
        log.record(
            LineageRecord(
                run_id="r1", node_id="backend", agent="backend", attempt=i,
                gate="schema_diff", verdict=verdict, cost_usd=0.01,
            )
        )
    log.record(LineageRecord(run_id="r2", node_id="x", agent="x", attempt=1))

    recs = log.read("r1")
    assert [r.attempt for r in recs] == [1, 2, 3]
    assert [r.verdict for r in recs] == ["rejected", "rejected", "approved"]
    assert recs[0].ts <= recs[1].ts <= recs[2].ts
    assert log.total_cost("r1") == pytest.approx(0.03)
    assert len(log.read()) == 4
