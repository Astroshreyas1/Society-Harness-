import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from society.memory import KVStore, LineageLog, LineageRecord
from society.providers import NodeInput, NodeOutput, Worker

from .deterministic import NoErrorGate
from .types import Gate, GateContext, GateResult


class EscalationRequired(Exception):
    def __init__(self, node_id: str, reason: str, history: list[GateResult]):
        super().__init__(f"{node_id}: {reason}")
        self.node_id = node_id
        self.reason = reason
        self.history = history


class BudgetExceeded(EscalationRequired):
    pass


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    max_run_cost_usd: float = 5.0


@dataclass
class NodeOutcome:
    node_id: str
    output: NodeOutput
    attempt: int
    gates: list[GateResult]
    approved: bool
    written_keys: list[str] = field(default_factory=list)


OutputExtractor = Callable[[NodeOutput], dict[str, Any]]
Observer = Callable[[str, dict[str, Any]], Awaitable[None] | None]


def _feedback_task(task: str, attempt: int, failures: list[GateResult]) -> str:
    lines = [f"[Retry — attempt {attempt}] Your previous attempt was rejected by the quality gate:"]
    for g in failures:
        lines.append(f"- {g.gate}: {g.reason}")
    lines.append(
        "Your workspace still contains the files from that attempt. Start by reading the files named in the "
        "failure (and the failing test) with read_file; then change only what the failure indicates, re-run "
        "the check yourself, and only then report. Do not rewrite everything from scratch and do not repeat "
        "the same mistake."
    )
    return f"{task}\n\n" + "\n".join(lines)


async def execute_node(
    run_id: str,
    node: NodeInput,
    worker: Worker,
    gates: list[Gate],
    kv: KVStore,
    lineage: LineageLog,
    policy: RetryPolicy = RetryPolicy(),
    outputs: dict[str, OutputExtractor] | None = None,
    observer: Observer | None = None,
) -> NodeOutcome:
    """Run one DAG node under the retry/gate/promote loop.

    `outputs` maps a dtype name -> extractor over NodeOutput; each is written as a
    versioned attempt to KV and promoted to :current only when every gate approves.
    """
    outputs = outputs or {}
    gates = [NoErrorGate(), *gates]
    history: list[GateResult] = []
    task = node.task

    async def emit(event: str, **data: Any) -> None:
        if observer:
            r = observer(event, {"run_id": run_id, "node_id": node.node_id, **data})
            if r is not None:
                await r

    for _ in range(policy.max_attempts):
        spent = lineage.total_cost(run_id)
        if spent >= policy.max_run_cost_usd:
            reason = f"run cost ${spent:.2f} exceeded ceiling ${policy.max_run_cost_usd:.2f}"
            lineage.record(LineageRecord(run_id=run_id, node_id=node.node_id, agent=node.agent, attempt=0,
                                         gate="budget", verdict="escalated", reason=reason))
            raise BudgetExceeded(node.node_id, reason, history)

        attempt = kv.next_attempt(run_id, node.agent, "_run")
        kv.put_attempt(run_id, node.agent, "_run", attempt, {"started": time.time()})
        attempt_node = node.model_copy(update={"task": task, "attempt": attempt, "run_id": run_id})
        await emit("node_start", attempt=attempt, agent=node.agent, provider=worker.name)

        started = time.time()
        output = await worker.run(attempt_node)
        finished = time.time()

        written: list[str] = []
        for dtype, extract in outputs.items():
            try:
                value = extract(output)
            except Exception as e:  # noqa: BLE001
                value = None
                output.error = output.error or f"extractor {dtype} failed: {e}"
            if value is not None:
                written.append(kv.put_attempt(run_id, node.agent, dtype, attempt, value))

        results: list[GateResult] = []
        ctx = GateContext(run_id=run_id, node=attempt_node, output=output, attempt=attempt,
                          kv=kv, workspace=node.workspace, prior=results)
        for gate in gates:
            res = await gate.check(ctx)
            results.append(res)
            await emit("gate", attempt=attempt, gate=res.gate, verdict=res.verdict,
                       confidence=res.confidence, reason=res.reason)
            if not res.approved:
                break
        history.extend(results)

        final = results[-1]
        lineage.record(LineageRecord(
            run_id=run_id, node_id=node.node_id, agent=node.agent, attempt=attempt,
            provider=output.provider, model=output.model, fallback_from=output.fallback_from,
            inputs_read=list(node.context.keys()), outputs_written=written,
            files_written=output.files_written, gate=final.gate, verdict=final.verdict,
            confidence=final.confidence, reason=final.reason,
            input_tokens=output.input_tokens, output_tokens=output.output_tokens,
            cost_usd=output.cost_usd, duration_ms=output.duration_ms,
            started_at=started, finished_at=finished, error=output.error,
            text_tail=(output.text or "")[-1500:],
        ))

        if final.approved:
            for dtype in outputs:
                if kv.exists(f"run:{run_id}:{node.agent}:{dtype}:attempt_{attempt}"):
                    kv.promote(run_id, node.agent, dtype, attempt)
            await emit("node_approved", attempt=attempt)
            return NodeOutcome(node.node_id, output, attempt, history, True, written)

        if final.verdict in ("abstained", "error"):
            reason = f"gate {final.gate} could not decide: {final.reason}"
            lineage.record(LineageRecord(run_id=run_id, node_id=node.node_id, agent=node.agent,
                                         attempt=attempt, gate=final.gate, verdict="escalated", reason=reason))
            await emit("escalated", attempt=attempt, reason=reason)
            raise EscalationRequired(node.node_id, reason, history)

        await emit("node_rejected", attempt=attempt, reason=final.reason)
        task = _feedback_task(node.task, attempt + 1, [r for r in results if not r.approved])

    reason = f"rejected {policy.max_attempts} times; last: {history[-1].reason}"
    lineage.record(LineageRecord(run_id=run_id, node_id=node.node_id, agent=node.agent,
                                 attempt=policy.max_attempts, gate=history[-1].gate,
                                 verdict="escalated", reason=reason))
    await emit("escalated", attempt=policy.max_attempts, reason=reason)
    raise EscalationRequired(node.node_id, reason, history)
