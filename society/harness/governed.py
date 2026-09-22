"""Where the harness attaches to the society: three seams, the agents' code untouched.

  GovernedWorker       wraps a role's worker: a model-tier unit (slot, request budget, token budget, tenant budget) is
                       acquired before the call and released with the realised usage after it. Under `off` an api-like
                       refusal is retried the way the client SDK would (SDK_RETRIES, exponential backoff honouring
                       retry-after), then reported as a provider error — the attempt is burned, as in the recordings.
  GovernedCommandGate  the deterministic test gates (`backend_tests`, `merged_tests`) take a sandbox CPU unit for the run.
  tool hook            the shell tools the replay scripts call (pytest inside a role's loop) take one too.
  GovernedRunner       registers every DAG run's service graph with the gateway when it starts and reports its end;
                       the executor's verdicts reach the gateway through the observer.
"""
from __future__ import annotations

import random
import uuid
from typing import Any, Callable

from society.gates import CommandGate, GateContext, GateResult
from society.gates.executor import Observer
from society.memory import KVStore, LineageLog
from society.providers import NodeInput, NodeOutput, Worker
from society.providers.replay import ProviderError, ReplayWorker
from society.runner import AgentSpec, DagRunner, DagSpec

from .gateway import SDK_RETRIES, Gateway, NodeRef


class GovernedWorker:
    """Admission around one role's model call."""

    def __init__(self, inner: Worker, gateway: Gateway, resource: str, tenant_of: Callable[[str], str]):
        self.inner, self.gw, self.resource, self.tenant_of = inner, gateway, resource, tenant_of
        self.name = getattr(inner, "name", "worker")
        self.model = getattr(inner, "model", "")
        self.provider_errors = 0

    def _error(self, node: NodeInput, e: ProviderError) -> NodeOutput:
        self.provider_errors += 1
        return NodeOutput(node_id=node.node_id, agent=node.agent, provider=self.name, model=self.model,
                          error=f"{e.kind}: {e}", error_kind="provider")

    async def run(self, node: NodeInput) -> NodeOutput:
        sid = f"{node.run_id}/{node.node_id}"
        tenant = self.tenant_of(node.run_id)
        delay = 0.5
        for k in range(SDK_RETRIES + 1):
            try:
                grant = await self.gw.acquire(sid, self.resource, "chat", run_id=node.run_id, node_id=node.node_id, agent=node.agent,
                                              attempt=node.attempt, tenant=tenant)
                break
            except ProviderError as e:
                if e.kind in ("budget", "timeout") or k == SDK_RETRIES:
                    return self._error(node, e)
                wait = e.retry_after if e.retry_after > 0 else delay * random.uniform(0.5, 1.5)
                delay = min(8.0, delay * 2)
                await self.gw.world.clock.sleep(wait)
        try:
            out = await self.inner.run(node)
        except Exception as e:  # noqa: BLE001 — the unit must go back whatever the worker did
            self.gw.release(grant, outcome="error")
            raise e
        rejected = out.error is not None or out.result is None
        try:
            self.gw.release(grant, tokens_in=out.input_tokens, tokens_out=out.output_tokens, outcome="ok", rejected=rejected,
                            reject_gate="no_error" if rejected else None)
        except ProviderError as e:                                          # the platform could not collect: the work is lost
            return self._error(node, e)
        return out


class GovernedCommandGate(CommandGate):
    def __init__(self, inner: CommandGate, gateway: Gateway, tenant_of: Callable[[str], str]):
        super().__init__(inner.command, inner.timeout, inner.name)
        self.gw, self.tenant_of = gateway, tenant_of

    async def check(self, ctx: GateContext) -> GateResult:
        sid = f"{ctx.run_id}/{ctx.node.node_id}"
        g = await self.gw.acquire(sid, "sandbox.cpu", f"gate:{self.name}", run_id=ctx.run_id, node_id=ctx.node.node_id, agent=ctx.node.agent,
                                  attempt=ctx.attempt, tenant=self.tenant_of(ctx.run_id))
        try:
            return await super().check(ctx)
        finally:
            self.gw.release(g)


def tool_hook_for(gateway: Gateway, tenant_of: Callable[[str], str], agent_of: Callable[[str], tuple[str, str, int]]):
    """The replay scripts' shell calls take a sandbox unit. `agent_of(sid)` recovers (run_id, node_id, attempt) for the session."""
    async def hook(sid: str, kind: str, fn: Callable[[], str]) -> str:
        run_id, node_id, attempt = agent_of(sid)
        agent = gateway.runs[run_id].nodes[node_id].agent if run_id in gateway.runs and node_id in gateway.runs[run_id].nodes else node_id
        return await gateway.tool(sid, kind, fn, run_id=run_id, node_id=node_id, agent=agent, attempt=attempt, tenant=tenant_of(run_id))
    return hook


class GovernedRunner(DagRunner):
    def __init__(self, agents: dict[str, AgentSpec], kv: KVStore, lineage: LineageLog, workspace_root, gateway: Gateway,
                 tenant_of: Callable[[str], str], providers: dict[str, str], **kw: Any):
        self.gw, self.tenant_of, self.providers = gateway, tenant_of, providers
        user_observer: Observer | None = kw.pop("observer", None)

        def observer(event: str, data: dict[str, Any]):
            if event in ("node_approved", "node_rejected"):
                self.gw.note_verdict(data["run_id"], data["node_id"], event == "node_approved")
            if user_observer is not None:
                return user_observer(event, data)
            return None

        super().__init__(agents, kv, lineage, workspace_root, observer=observer, **kw)

    async def run(self, dag: DagSpec, run_id: str | None = None):
        run_id = run_id or uuid.uuid4().hex[:8]
        nodes = [NodeRef(n.id, n.agent, self.gw.world.resource_of(self.providers[n.agent]), list(n.depends_on),
                         paid=self.gw.world.resource_of(self.providers[n.agent]) in self.gw.world.prices) for n in dag.nodes]
        await self.gw.admit_run(run_id, self.tenant_of(run_id), nodes)    # workflow-level pacing: deferred, never burned
        self.gw.run_start(run_id, self.tenant_of(run_id), nodes)
        status = "error"
        try:
            result = await super().run(dag, run_id=run_id)
            status = result.status
        finally:
            self.gw.run_end(run_id, status)
        return result


def govern(agents: dict[str, AgentSpec], gateway: Gateway, tenant_of: Callable[[str], str]) -> dict[str, AgentSpec]:
    """Wrap every role's worker and every command gate; the replay scripts' tool hook is set on the workers."""
    attempts: dict[str, int] = {}

    def agent_of(sid: str) -> tuple[str, str, int]:
        run_id, _, node_id = sid.partition("/")
        run = gateway.runs.get(run_id)
        attempt = run.attempts.get(node_id, 1) if run is not None else 1
        return run_id, node_id, attempt

    hook = tool_hook_for(gateway, tenant_of, agent_of)
    for spec in agents.values():
        inner = spec.worker
        if isinstance(inner, ReplayWorker):
            inner.tool_hook = hook
            res = gateway.world.resource_of(inner.provider)
        else:
            res = gateway.world.resource_of(getattr(inner, "provider", "anthropic"))
        spec.worker = GovernedWorker(inner, gateway, res, tenant_of)
        spec.gates = [GovernedCommandGate(g, gateway, tenant_of) if isinstance(g, CommandGate) else g for g in spec.gates]
    return agents
