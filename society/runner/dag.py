import asyncio
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from society.gates import EscalationRequired, NodeOutcome, RetryPolicy, execute_node, validate_dag
from society.gates.executor import Observer
from society.memory import KVStore, LineageLog
from society.providers import NodeInput

from .spec import AgentSpec, DagSpec, NodeSpec, parse_ref


@dataclass
class RunResult:
    run_id: str
    status: Literal["completed", "escalated", "invalid"]
    outcomes: dict[str, NodeOutcome] = field(default_factory=dict)
    escalation: EscalationRequired | None = None
    reason: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at


class DagRunner:
    def __init__(
        self,
        agents: dict[str, AgentSpec],
        kv: KVStore,
        lineage: LineageLog,
        workspace_root: Path,
        policy: RetryPolicy = RetryPolicy(),
        observer: Observer | None = None,
    ):
        self.agents = agents
        self.kv = kv
        self.lineage = lineage
        self.workspace_root = Path(workspace_root)
        self.policy = policy
        self.observer = observer

    async def _emit(self, event: str, **data: Any) -> None:
        if self.observer:
            r = self.observer(event, data)
            if r is not None:
                await r

    def _validate(self, dag: DagSpec) -> str | None:
        res = validate_dag(dag.as_dict())
        if not res.approved:
            return res.reason
        for n in dag.nodes:
            if n.agent not in self.agents:
                return f"node {n.id} uses unknown agent {n.agent!r}"
            for ref in n.inputs + n.outputs:
                parse_ref(ref)
            for ref in n.outputs:
                agent, dtype = parse_ref(ref)
                if agent != n.agent:
                    return f"node {n.id} may only publish under its own agent, not {ref!r}"
                if dtype not in self.agents[n.agent].outputs:
                    return f"agent {n.agent!r} has no extractor for output {dtype!r}"
        return None

    def _build_input(self, run_id: str, run_dir: Path, node: NodeSpec, dag: DagSpec) -> NodeInput:
        agent = self.agents[node.agent]
        context: dict[str, Any] = {}
        for ref in node.inputs:
            a, d = parse_ref(ref)
            value = self.kv.get_current(run_id, a, d)
            if value is None:
                raise RuntimeError(f"{node.id}: upstream {ref} has no :current (gate never promoted it)")
            context[ref] = value

        ws_rel = node.workspace if node.workspace is not None else node.agent
        workspace = (run_dir / ws_rel).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        if agent.seed and not any(workspace.iterdir()):
            shutil.copytree(agent.seed, workspace, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        if ws_rel in (".", ""):
            by_id = {n.id: n for n in dag.nodes}
            upstream = sorted({by_id[d].workspace or by_id[d].agent for d in node.depends_on})
            context["upstream_workspaces"] = {
                "note": "Upstream agents worked in these subdirectories of your workspace.",
                "dirs": upstream,
            }
        if agent.prepare and not self.kv.exists(f"run:{run_id}:{node.agent}:_prepared"):
            extra = agent.prepare(workspace, context)
            self.kv.put(f"run:{run_id}:{node.agent}:_prepared", True)
            if extra:
                context.update(extra)

        return NodeInput(
            node_id=node.id,
            agent=node.agent,
            system_prompt=agent.system_prompt,
            task=node.task,
            context=context,
            tools=node.tools if node.tools is not None else agent.tools,
            workspace=workspace,
            output_schema=agent.output_schema,
        )

    async def run(self, dag: DagSpec, run_id: str | None = None) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:8]
        result = RunResult(run_id=run_id, status="completed", started_at=time.time())
        run_dir = self.workspace_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if (err := self._validate(dag)) is not None:
            result.status, result.reason, result.finished_at = "invalid", err, time.time()
            await self._emit("run_invalid", run_id=run_id, reason=err)
            return result

        await self._emit("run_start", run_id=run_id, goal=dag.goal, nodes=[n.id for n in dag.nodes])
        done: dict[str, asyncio.Event] = {n.id: asyncio.Event() for n in dag.nodes}
        failed = asyncio.Event()
        by_id = {n.id: n for n in dag.nodes}

        async def run_node(node: NodeSpec) -> None:
            for dep in node.depends_on:
                await done[dep].wait()
                if failed.is_set():
                    return
            if failed.is_set():
                return
            agent = self.agents[node.agent]
            outputs = {parse_ref(r)[1]: agent.outputs[parse_ref(r)[1]] for r in node.outputs}
            try:
                node_input = self._build_input(run_id, run_dir, node, dag)
                outcome = await execute_node(
                    run_id, node_input, agent.worker, agent.gates, self.kv, self.lineage,
                    policy=self.policy, outputs=outputs, observer=self.observer,
                )
                result.outcomes[node.id] = outcome
            except (EscalationRequired, RuntimeError) as e:
                result.status, result.reason = "escalated", str(e)
                result.escalation = e if isinstance(e, EscalationRequired) else None
                failed.set()
                for ev in done.values():
                    ev.set()
                return
            done[node.id].set()

        await asyncio.gather(*(run_node(n) for n in dag.nodes))
        result.finished_at = time.time()
        await self._emit("run_end", run_id=run_id, status=result.status, reason=result.reason,
                         duration_s=round(result.duration_s, 2), cost_usd=round(self.lineage.total_cost(run_id), 4))
        return result
