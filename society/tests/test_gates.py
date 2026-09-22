from pathlib import Path
from typing import Any

import pytest

from society.gates import (
    CommandGate,
    DecisionConsistencyGate,
    EscalationRequired,
    BudgetExceeded,
    GateContext,
    GateResult,
    JevGate,
    OutputSchemaGate,
    RetryPolicy,
    SchemaDiffGate,
    diff_schemas,
    execute_node,
    validate_dag,
)
from society.memory import KVStore, LineageLog
from society.providers import NodeInput, NodeOutput

LOGIN_SCHEMA = {
    "type": "object",
    "properties": {"token": {"type": "string"}, "userId": {"type": "integer"}},
    "required": ["token", "userId"],
}


def _node(tmp_path: Path, **kw: Any) -> NodeInput:
    base = dict(node_id="n1", agent="frontend", system_prompt="s", task="t", tools=[], workspace=tmp_path)
    base.update(kw)
    return NodeInput(**base)


def _ctx(tmp_path: Path, output: NodeOutput, kv: KVStore | None = None, node: NodeInput | None = None) -> GateContext:
    return GateContext(run_id="r1", node=node or _node(tmp_path), output=output, attempt=1,
                       kv=kv or KVStore(), workspace=tmp_path)


def _out(**kw: Any) -> NodeOutput:
    base = dict(node_id="n1", agent="frontend", provider="stub", model="stub")
    base.update(kw)
    return NodeOutput(**base)


# --- DAG validator -----------------------------------------------------------

GOOD_DAG = {"nodes": [
    {"id": "backend", "agent": "backend", "outputs": ["backend:schema"]},
    {"id": "frontend", "agent": "frontend", "depends_on": ["backend"], "inputs": ["backend:schema"], "outputs": ["frontend:ui"]},
    {"id": "integration", "agent": "integration", "depends_on": ["backend", "frontend"], "inputs": ["backend:schema", "frontend:ui"]},
]}


def test_dag_good():
    r = validate_dag(GOOD_DAG)
    assert r.approved, r.reason
    assert r.details["order"][0] == "backend"


def test_dag_cycle():
    dag = {"nodes": [{"id": "a", "agent": "x", "depends_on": ["b"]}, {"id": "b", "agent": "x", "depends_on": ["a"]}]}
    assert "cycle" in validate_dag(dag).reason


def test_dag_unknown_dep_and_unproduced_input():
    assert "unknown node" in validate_dag({"nodes": [{"id": "a", "agent": "x", "depends_on": ["zz"]}]}).reason
    dag = {"nodes": [{"id": "a", "agent": "x", "inputs": ["nobody:makes_this"]}]}
    assert "no upstream node produces" in validate_dag(dag).reason


# --- Schema diff -------------------------------------------------------------

def test_diff_schemas_reports_field_mismatch():
    used = {"type": "object", "properties": {"authToken": {"type": "string"}, "user_id": {"type": "integer"}},
            "required": ["authToken", "user_id"]}
    diffs = diff_schemas(LOGIN_SCHEMA, used)
    assert any("token" in d and "missing" in d for d in diffs)
    assert any("authToken" in d for d in diffs)
    assert diff_schemas(LOGIN_SCHEMA, LOGIN_SCHEMA) == []


async def test_schema_diff_gate(tmp_path: Path):
    kv = KVStore()
    kv.put_attempt("r1", "backend", "schema", 1, LOGIN_SCHEMA)
    gate = SchemaDiffGate("backend", "schema")

    r = await gate.check(_ctx(tmp_path, _out(result={"schema_used": LOGIN_SCHEMA}), kv))
    assert not r.approved and "no :current" in r.reason, "must not read an unpromoted attempt"

    kv.promote("r1", "backend", "schema", 1)
    assert (await gate.check(_ctx(tmp_path, _out(result={"schema_used": LOGIN_SCHEMA}), kv))).approved

    wrong = {"type": "object", "properties": {"token": {"type": "string"}}, "required": ["token"]}
    r = await gate.check(_ctx(tmp_path, _out(result={"schema_used": wrong}), kv))
    assert not r.approved and "userId" in r.reason


# --- Output schema / command -------------------------------------------------

async def test_output_schema_gate(tmp_path: Path):
    node = _node(tmp_path, output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
    assert (await OutputSchemaGate().check(_ctx(tmp_path, _out(result={"ok": True}), node=node))).approved
    r = await OutputSchemaGate().check(_ctx(tmp_path, _out(result={"ok": "yes"}), node=node))
    assert not r.approved and "violation" in r.reason
    assert not (await OutputSchemaGate().check(_ctx(tmp_path, _out(result=None), node=node))).approved


async def test_command_gate(tmp_path: Path):
    assert (await CommandGate("true").check(_ctx(tmp_path, _out()))).approved
    r = await CommandGate("echo boom >&2; exit 3").check(_ctx(tmp_path, _out()))
    assert not r.approved and "exit code 3" in r.reason and "boom" in r.reason


async def test_decision_consistency_gate(tmp_path: Path):
    g = DecisionConsistencyGate()
    ok = lambda go, items: _out(result={"go": go, "checklist": items, "summary": ""})
    assert (await g.check(_ctx(tmp_path, ok(True, [{"item": "tests", "ok": True}])))).approved
    assert (await g.check(_ctx(tmp_path, ok(False, [{"item": "tests", "ok": False}])))).approved
    r = await g.check(_ctx(tmp_path, ok(True, [{"item": "tests", "ok": True}, {"item": "risk", "ok": False}])))
    assert not r.approved and "risk" in r.reason
    assert not (await g.check(_ctx(tmp_path, ok(False, [{"item": "tests", "ok": True}])))).approved
    assert not (await g.check(_ctx(tmp_path, _out(result={"go": True})))).approved


# --- Retry / promote / escalate ---------------------------------------------

class FlakyWorker:
    """Fails `fail_times` attempts, then succeeds. Records the tasks it received."""

    name = "stub"

    def __init__(self, fail_times: int, cost: float = 0.1):
        self.fail_times = fail_times
        self.cost = cost
        self.calls = 0
        self.tasks: list[str] = []

    async def run(self, node: NodeInput) -> NodeOutput:
        self.calls += 1
        self.tasks.append(node.task)
        ok = self.calls > self.fail_times
        return _out(agent=node.agent, result={"ok": ok, "schema": LOGIN_SCHEMA if ok else None},
                    cost_usd=self.cost, duration_ms=1)


class ResultOkGate:
    name = "result_ok"

    async def check(self, ctx: GateContext) -> GateResult:
        ok = bool((ctx.output.result or {}).get("ok"))
        return GateResult(gate=self.name, verdict="approved" if ok else "rejected",
                          reason="ok" if ok else "result.ok was false")


class AbstainGate:
    name = "abstain"

    async def check(self, ctx: GateContext) -> GateResult:
        return GateResult(gate=self.name, verdict="abstained", confidence=0.2, reason="unsure")


async def test_retry_then_promote(tmp_path: Path):
    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")
    worker = FlakyWorker(fail_times=2)
    events: list[tuple[str, dict]] = []

    outcome = await execute_node(
        "r1", _node(tmp_path, agent="backend"), worker, [ResultOkGate()], kv, log,
        outputs={"schema": lambda o: (o.result or {}).get("schema")},
        observer=lambda e, d: events.append((e, d)),
    )

    assert outcome.approved and outcome.attempt == 3 and worker.calls == 3
    assert kv.get_current("r1", "backend", "schema") == LOGIN_SCHEMA
    assert kv.current_attempt("r1", "backend", "schema") == 3
    assert kv.get_attempt("r1", "backend", "schema", 1) is None, "failed attempts had no schema to write"
    assert "[Retry — attempt 2]" in worker.tasks[1] and "result.ok was false" in worker.tasks[1]
    assert "[Retry" not in worker.tasks[0]

    verdicts = [r.verdict for r in log.read("r1")]
    assert verdicts == ["rejected", "rejected", "approved"]
    assert [e for e, _ in events].count("node_rejected") == 2
    assert events[-1][0] == "node_approved"


async def test_escalates_after_max_attempts(tmp_path: Path):
    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")
    worker = FlakyWorker(fail_times=99)
    with pytest.raises(EscalationRequired) as ei:
        await execute_node("r1", _node(tmp_path, agent="backend"), worker, [ResultOkGate()], kv, log,
                           policy=RetryPolicy(max_attempts=3),
                           outputs={"schema": lambda o: (o.result or {}).get("schema")})
    assert worker.calls == 3
    assert "3 times" in ei.value.reason
    assert kv.get_current("r1", "backend", "schema") is None
    assert log.read("r1")[-1].verdict == "escalated"


async def test_abstain_escalates_immediately(tmp_path: Path):
    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")
    worker = FlakyWorker(fail_times=0)
    with pytest.raises(EscalationRequired) as ei:
        await execute_node("r1", _node(tmp_path), worker, [AbstainGate()], kv, log)
    assert worker.calls == 1 and "could not decide" in ei.value.reason


async def test_worker_error_is_rejected(tmp_path: Path):
    class Broken:
        name = "stub"

        async def run(self, node: NodeInput) -> NodeOutput:
            return _out(error="boom")

    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")
    with pytest.raises(EscalationRequired):
        await execute_node("r1", _node(tmp_path), Broken(), [], kv, log, policy=RetryPolicy(max_attempts=2))
    assert all(r.gate == "no_error" for r in log.read("r1") if r.verdict == "rejected")


async def test_budget_ceiling(tmp_path: Path):
    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")
    worker = FlakyWorker(fail_times=99, cost=1.0)
    with pytest.raises(BudgetExceeded):
        await execute_node("r1", _node(tmp_path), worker, [ResultOkGate()], kv, log,
                           policy=RetryPolicy(max_attempts=10, max_run_cost_usd=2.5))
    assert worker.calls == 3, "third attempt spent $3 > $2.5, fourth must not start"


# --- Jev gate: the rule stand-in (no model needed) -------------------------------------------

async def test_jev_rule_stand_in_judges_against_gate_evidence(tmp_path: Path):
    """The rule backend: clear contradictions are confident rejections; consistent reports pass; it names itself a rule."""
    shipn = _node(tmp_path, agent="ship", task="Verify the merged app and produce the go/no-go decision.")
    intn = _node(tmp_path, agent="integration", task="Run the merged tests and report.")
    gate = JevGate(criteria="Each checklist note must support its ok value; the summary must not contradict the checklist.", backend="rule")
    assert gate.name == "jev_rule" and gate.backend == "rule"
    merged_ok = GateResult(gate="merged_tests", verdict="approved", reason="exit code 0")
    merged_bad = GateResult(gate="merged_tests", verdict="rejected", reason="exit code 1")
    cases = [
        ("ship-ok", shipn, _out(result={"go": True, "checklist": [{"item": "tests", "ok": True, "note": "pytest green"}], "summary": "ready to ship"}), [], "approved"),
        ("ship-note-contradicts", shipn, _out(result={"go": True, "checklist": [{"item": "tests", "ok": True, "note": "3 tests failing"}], "summary": "ship"}), [], "rejected"),
        ("ship-summary-contradicts", shipn, _out(result={"go": True, "checklist": [{"item": "tests", "ok": True, "note": "fine"}], "summary": "blocked, do not ship"}), [], "rejected"),
        ("int-consistent", intn, _out(result={"merged_files": ["app/main.py"], "tests_passed": True, "summary": "tests pass"}), [merged_ok], "approved"),
        ("int-claims-pass-but-failed", intn, _out(result={"merged_files": ["app/main.py"], "tests_passed": True, "summary": "tests pass"}), [merged_bad], "rejected"),
        ("int-merge-skipped", intn, _out(result={"merged_files": [], "tests_passed": True, "summary": "merge skipped"}), [merged_ok], "rejected"),
    ]
    for name, node, output, prior, expected in cases:
        r = await gate.check(GateContext(run_id="r1", node=node, output=output, attempt=1, kv=KVStore(), workspace=tmp_path, prior=prior))
        assert r.verdict == expected, f"{name}: got {r.verdict} — {r.reason}"
        assert r.confidence >= 0.6 and r.details.get("rule") is True


def test_jev_gate_falls_back_to_rule_without_a_key(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    g = JevGate(criteria="x", backend="remote")
    assert g.backend == "rule" and g.name == "jev_rule"
    with pytest.raises(ValueError):
        JevGate(criteria="x", backend="ollama")


# --- "did you actually do it" gates ------------------------------------------

async def test_files_written_and_endpoint_present(tmp_path: Path):
    from society.gates import EndpointPresentGate, FilesWrittenGate, UsesEndpointGate
    (tmp_path / "app").mkdir(); (tmp_path / "static").mkdir()
    (tmp_path / "app" / "main.py").write_text('@app.post("/login")\ndef login(): ...\n')
    (tmp_path / "static" / "index.html").write_text('fetch("/login", {method: "POST"})')

    lied = _out(files_written=[], result={"endpoint": "POST /login"})
    r = await FilesWrittenGate(["app/", "tests/"]).check(_ctx(tmp_path, lied))
    assert not r.approved and "reported but not done" in r.reason

    did = _out(files_written=["app/main.py", "tests/test_login.py"], result={"endpoint": "POST /login"})
    assert (await FilesWrittenGate(["app/", "tests/"]).check(_ctx(tmp_path, did))).approved
    assert (await EndpointPresentGate().check(_ctx(tmp_path, did))).approved
    r = await EndpointPresentGate().check(_ctx(tmp_path, _out(result={"endpoint": "POST /logout"})))
    assert not r.approved and "/logout" in r.reason

    kv = KVStore(); kv.put_attempt("r1", "backend", "endpoint", 1, "POST /login"); kv.promote("r1", "backend", "endpoint", 1)
    assert (await UsesEndpointGate("backend").check(_ctx(tmp_path, _out(), kv))).approved
    (tmp_path / "static" / "index.html").write_text('fetch("/auth")')
    assert not (await UsesEndpointGate("backend").check(_ctx(tmp_path, _out(), kv))).approved
