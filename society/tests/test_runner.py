import asyncio
from pathlib import Path
from typing import Any

from society.gates import GateContext, GateResult, RetryPolicy
from society.memory import KVStore, LineageLog
from society.providers import NodeInput, NodeOutput
from society.runner import AgentSpec, DagRunner, DagSpec, NodeSpec, Planner


class SleepWorker:
    """Sleeps, then publishes {agent: <name>, saw: <context keys>}. Records what it saw."""

    name = "stub"

    def __init__(self, delay: float = 0.2, fail_first: bool = False):
        self.delay = delay
        self.fail_first = fail_first
        self.calls = 0
        self.seen: list[NodeInput] = []

    async def run(self, node: NodeInput) -> NodeOutput:
        self.calls += 1
        self.seen.append(node)
        await asyncio.sleep(self.delay)
        (node.workspace / f"{node.agent}.txt").write_text("done")
        ok = not (self.fail_first and self.calls == 1)
        return NodeOutput(node_id=node.node_id, agent=node.agent, provider="stub", model="stub",
                          result={"ok": ok, "payload": {"from": node.agent, "saw": sorted(node.context)}},
                          files_written=[f"{node.agent}.txt"], duration_ms=int(self.delay * 1000))


class OkGate:
    name = "ok"

    async def check(self, ctx: GateContext) -> GateResult:
        ok = bool((ctx.output.result or {}).get("ok"))
        return GateResult(gate=self.name, verdict="approved" if ok else "rejected", reason="ok" if ok else "not ok")


def _agents(**workers: SleepWorker) -> dict[str, AgentSpec]:
    return {
        name: AgentSpec(name=name, worker=w, system_prompt=f"you are {name}", gates=[OkGate()],
                        outputs={"payload": lambda o: (o.result or {}).get("payload")})
        for name, w in workers.items()
    }


DAG = DagSpec(goal="A and B in parallel, then C", nodes=[
    NodeSpec(id="a", agent="a", task="do a", outputs=["a:payload"]),
    NodeSpec(id="b", agent="b", task="do b", outputs=["b:payload"]),
    NodeSpec(id="c", agent="c", task="merge", depends_on=["a", "b"], inputs=["a:payload", "b:payload"],
             outputs=["c:payload"], workspace="."),
])


async def test_parallel_then_barrier(tmp_path: Path):
    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")
    a, b, c = SleepWorker(0.3), SleepWorker(0.3), SleepWorker(0.05)
    events: list[tuple[str, dict[str, Any]]] = []
    runner = DagRunner(_agents(a=a, b=b, c=c), kv, log, tmp_path / "ws",
                       observer=lambda e, d: events.append((e, d)))

    res = await runner.run(DAG, run_id="r1")

    assert res.status == "completed", res.reason
    assert set(res.outcomes) == {"a", "b", "c"}
    assert res.duration_s < 0.55, f"A and B did not overlap: {res.duration_s:.2f}s"

    recs = {r.node_id: r for r in log.read("r1")}
    assert recs["a"].started_at < recs["b"].finished_at and recs["b"].started_at < recs["a"].finished_at
    assert recs["c"].started_at >= max(recs["a"].finished_at, recs["b"].finished_at)

    assert c.seen[0].context["a:payload"] == {"from": "a", "saw": []}
    assert c.seen[0].context["b:payload"] == {"from": "b", "saw": []}
    assert c.seen[0].context["upstream_workspaces"]["dirs"] == ["a", "b"]
    assert sorted(a.seen[0].context) == [], "A must start with an empty context"

    assert (tmp_path / "ws/r1/a/a.txt").exists() and (tmp_path / "ws/r1/b/b.txt").exists()
    assert (tmp_path / "ws/r1/c.txt").exists(), "C works in the run root"
    assert kv.get_current("r1", "c", "payload")["saw"] == ["a:payload", "b:payload", "upstream_workspaces"]
    assert [e for e, _ in events][0] == "run_start" and events[-1][0] == "run_end"


async def test_retry_upstream_does_not_leak_downstream(tmp_path: Path):
    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")
    a, b, c = SleepWorker(0.05, fail_first=True), SleepWorker(0.05), SleepWorker(0.01)
    runner = DagRunner(_agents(a=a, b=b, c=c), kv, log, tmp_path / "ws")

    res = await runner.run(DAG, run_id="r1")

    assert res.status == "completed"
    assert a.calls == 2 and res.outcomes["a"].attempt == 2
    assert "[Retry" in a.seen[1].task and "not ok" in a.seen[1].task
    assert kv.current_attempt("r1", "a", "payload") == 2
    assert c.calls == 1, "C must wait for A's approved attempt, not run on the rejected one"


async def test_escalation_stops_downstream(tmp_path: Path):
    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")

    class AlwaysBad(SleepWorker):
        async def run(self, node: NodeInput) -> NodeOutput:
            out = await super().run(node)
            out.result["ok"] = False
            return out

    a, b, c = AlwaysBad(0.05), SleepWorker(0.05), SleepWorker(0.01)
    runner = DagRunner(_agents(a=a, b=b, c=c), kv, log, tmp_path / "ws", policy=RetryPolicy(max_attempts=2))

    res = await runner.run(DAG, run_id="r1")

    assert res.status == "escalated" and res.escalation is not None
    assert res.escalation.node_id == "a" and a.calls == 2
    assert c.calls == 0, "C must never run when an upstream escalates"
    assert "b" in res.outcomes, "B was independent and may have finished"


async def test_invalid_dag_is_refused_before_any_work(tmp_path: Path):
    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")
    a = SleepWorker(0.01)
    runner = DagRunner(_agents(a=a), kv, log, tmp_path / "ws")

    bad = DagSpec(goal="x", nodes=[NodeSpec(id="a", agent="a", task="t", outputs=["b:payload"])])
    res = await runner.run(bad, run_id="r1")
    assert res.status == "invalid" and "own agent" in res.reason and a.calls == 0

    bad2 = DagSpec(goal="x", nodes=[NodeSpec(id="a", agent="ghost", task="t")])
    assert (await runner.run(bad2, run_id="r2")).status == "invalid"


async def test_planner_validates_structured_plan(tmp_path: Path):
    class PlanWorker:
        name = "stub"

        def __init__(self, plan: dict[str, Any]):
            self.plan = plan

        async def run(self, node: NodeInput) -> NodeOutput:
            assert node.output_schema is not None and "nodes" in node.system_prompt or True
            return NodeOutput(node_id=node.node_id, agent="planner", provider="stub", model="stub", result=self.plan)

    agents = _agents(a=SleepWorker(), b=SleepWorker())
    good = {"goal": "g", "nodes": [
        {"id": "a", "agent": "a", "task": "t", "depends_on": [], "inputs": [], "outputs": ["a:payload"]},
        {"id": "b", "agent": "b", "task": "t", "depends_on": ["a"], "inputs": ["a:payload"], "outputs": []},
    ]}
    dag, reason = await Planner(PlanWorker(good), agents, {"a": "does a", "b": "does b"}, tmp_path).plan("g")
    assert dag is not None and [n.id for n in dag.nodes] == ["a", "b"], reason

    cyclic = {"goal": "g", "nodes": [
        {"id": "a", "agent": "a", "task": "t", "depends_on": ["b"], "inputs": [], "outputs": []},
        {"id": "b", "agent": "b", "task": "t", "depends_on": ["a"], "inputs": [], "outputs": []},
    ]}
    dag, reason = await Planner(PlanWorker(cyclic), agents, {}, tmp_path).plan("g")
    assert dag is None and "cycle" in reason
