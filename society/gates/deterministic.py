import asyncio
import os
import re
import subprocess
from collections import deque
from typing import Any

import jsonschema

from .types import GateContext, GateResult


def validate_dag(dag: dict[str, Any], initial_keys: set[str] | None = None) -> GateResult:
    """Structural check: unique ids, deps exist, acyclic, every input has a producer."""
    name = "dag_validator"
    nodes = dag.get("nodes") or []
    if not nodes:
        return GateResult(gate=name, verdict="rejected", reason="dag has no nodes")

    ids = [n.get("id") for n in nodes]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        return GateResult(gate=name, verdict="rejected", reason="node ids must be unique and non-empty")

    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        for dep in n.get("depends_on", []):
            if dep not in by_id:
                return GateResult(gate=name, verdict="rejected", reason=f"{n['id']} depends on unknown node {dep}")
        if not n.get("agent"):
            return GateResult(gate=name, verdict="rejected", reason=f"{n['id']} has no agent")

    indeg = {i: len(by_id[i].get("depends_on", [])) for i in ids}
    children: dict[str, list[str]] = {i: [] for i in ids}
    for n in nodes:
        for dep in n.get("depends_on", []):
            children[dep].append(n["id"])
    order: list[str] = []
    q = deque(i for i in ids if indeg[i] == 0)
    while q:
        cur = q.popleft()
        order.append(cur)
        for c in children[cur]:
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) != len(ids):
        return GateResult(gate=name, verdict="rejected", reason="dag contains a cycle")

    produced: set[str] = set(initial_keys or set())
    for nid in order:
        n = by_id[nid]
        upstream = _ancestors(nid, by_id)
        upstream_outputs = {o for u in upstream for o in by_id[u].get("outputs", [])} | produced
        for inp in n.get("inputs", []):
            if inp not in upstream_outputs:
                return GateResult(
                    gate=name, verdict="rejected",
                    reason=f"{nid} reads '{inp}' which no upstream node produces",
                )
    return GateResult(gate=name, verdict="approved", reason="dag is acyclic and fully wired",
                      details={"order": order})


def _ancestors(nid: str, by_id: dict[str, Any]) -> set[str]:
    seen: set[str] = set()
    stack = list(by_id[nid].get("depends_on", []))
    while stack:
        cur = stack.pop()
        if cur not in seen:
            seen.add(cur)
            stack.extend(by_id[cur].get("depends_on", []))
    return seen


def diff_schemas(expected: dict[str, Any], actual: dict[str, Any], path: str = "$") -> list[str]:
    """Structural diff of two JSON Schemas: type, properties, required. Returns human-readable mismatches."""
    diffs: list[str] = []
    if expected.get("type") != actual.get("type"):
        diffs.append(f"{path}: type {expected.get('type')!r} != {actual.get('type')!r}")
    exp_req, act_req = set(expected.get("required", [])), set(actual.get("required", []))
    for missing in sorted(exp_req - act_req):
        diffs.append(f"{path}: required field '{missing}' missing")
    for extra in sorted(act_req - exp_req):
        diffs.append(f"{path}: unexpected required field '{extra}'")
    exp_props, act_props = expected.get("properties", {}), actual.get("properties", {})
    for k in sorted(set(exp_props) | set(act_props)):
        if k not in act_props:
            diffs.append(f"{path}.{k}: missing property")
        elif k not in exp_props:
            diffs.append(f"{path}.{k}: unexpected property")
        else:
            diffs.extend(diff_schemas(exp_props[k], act_props[k], f"{path}.{k}"))
    if "items" in expected or "items" in actual:
        if "items" in expected and "items" in actual:
            diffs.extend(diff_schemas(expected["items"], actual["items"], f"{path}[]"))
        else:
            diffs.append(f"{path}: items mismatch")
    return diffs


class SchemaDiffGate:
    """Compares the schema a consumer used against the schema the producer published in KV."""

    name = "schema_diff"

    def __init__(self, producer_agent: str, producer_dtype: str, consumer_result_key: str = "schema_used"):
        self.producer_agent = producer_agent
        self.producer_dtype = producer_dtype
        self.consumer_result_key = consumer_result_key

    async def check(self, ctx: GateContext) -> GateResult:
        expected = ctx.store.get_current(ctx.run_id, self.producer_agent, self.producer_dtype)
        if expected is None:
            return GateResult(gate=self.name, verdict="rejected",
                              reason=f"no :current {self.producer_agent}:{self.producer_dtype} in KV")
        actual = (ctx.output.result or {}).get(self.consumer_result_key)
        if not isinstance(actual, dict):
            return GateResult(gate=self.name, verdict="rejected",
                              reason=f"consumer output missing '{self.consumer_result_key}'")
        for label, s in (("published", expected), ("used", actual)):
            try:
                jsonschema.Draft202012Validator.check_schema(s)
            except jsonschema.SchemaError as e:
                return GateResult(gate=self.name, verdict="rejected", reason=f"{label} schema invalid: {e.message}")
        diffs = diff_schemas(expected, actual)
        if diffs:
            return GateResult(gate=self.name, verdict="rejected",
                              reason="schema mismatch: " + "; ".join(diffs), details={"diffs": diffs})
        return GateResult(gate=self.name, verdict="approved", reason="consumer schema matches published schema")


class OutputSchemaGate:
    """Validates the node's structured result against its declared output_schema."""

    name = "output_schema"

    async def check(self, ctx: GateContext) -> GateResult:
        if ctx.node.output_schema is None:
            return GateResult(gate=self.name, verdict="approved", reason="no output schema declared")
        if ctx.output.result is None:
            return GateResult(gate=self.name, verdict="rejected", reason="node produced no structured result")
        try:
            jsonschema.validate(ctx.output.result, ctx.node.output_schema)
        except jsonschema.ValidationError as e:
            return GateResult(gate=self.name, verdict="rejected", reason=f"output schema violation: {e.message}")
        return GateResult(gate=self.name, verdict="approved", reason="structured output valid")


class CommandGate:
    """Runs a command in the workspace; exit code 0 approves."""

    name = "command"

    def __init__(self, command: str, timeout: int = 300, name: str | None = None):
        self.command = command
        self.timeout = timeout
        if name:
            self.name = name

    async def check(self, ctx: GateContext) -> GateResult:
        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(self.command, shell=True, cwd=ctx.workspace,
                                  capture_output=True, text=True, timeout=self.timeout,
                                  env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})

        try:
            r = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            return GateResult(gate=self.name, verdict="rejected", reason=f"command timed out after {self.timeout}s")
        tail = (r.stdout + r.stderr)[-3000:]
        if r.returncode == 0:
            return GateResult(gate=self.name, verdict="approved", reason="exit code 0", details={"output": tail})
        return GateResult(gate=self.name, verdict="rejected",
                          reason=f"exit code {r.returncode}\n{tail}", details={"output": tail})


class NoErrorGate:
    """Rejects if the worker itself errored (SDK failure, budget, malformed JSON)."""

    name = "no_error"

    async def check(self, ctx: GateContext) -> GateResult:
        if ctx.output.error:
            return GateResult(gate=self.name, verdict="rejected", reason=ctx.output.error)
        return GateResult(gate=self.name, verdict="approved", reason="worker completed")


class DecisionConsistencyGate:
    """go=true requires every checklist item ok=true; go=false requires at least one ok=false."""

    name = "decision_consistency"

    async def check(self, ctx: GateContext) -> GateResult:
        r = ctx.output.result or {}
        go, items = r.get("go"), r.get("checklist") or []
        if not isinstance(go, bool) or not items:
            return GateResult(gate=self.name, verdict="rejected", reason="decision missing go or checklist")
        failed = [i.get("item", "?") for i in items if not i.get("ok")]
        if go and failed:
            return GateResult(gate=self.name, verdict="rejected",
                              reason=f"go=true but checklist items failed: {', '.join(failed)}. "
                                     "Either mark non-blocking findings ok=true (keep the note) or set go=false.")
        if not go and not failed:
            return GateResult(gate=self.name, verdict="rejected",
                              reason="go=false but every checklist item is ok=true; cite the blocking item")
        return GateResult(gate=self.name, verdict="approved", reason="decision consistent with checklist")


class FilesWrittenGate:
    """The worker must have actually changed files under each required prefix."""

    name = "files_written"

    def __init__(self, required_prefixes: list[str]):
        self.required = required_prefixes

    async def check(self, ctx: GateContext) -> GateResult:
        written = ctx.output.files_written
        missing = [p for p in self.required if not any(f.startswith(p) for f in written)]
        if missing:
            return GateResult(gate=self.name, verdict="rejected",
                              reason=f"no files written under {', '.join(missing)} — the work was reported but not done "
                                     f"(files written: {written or 'none'})")
        return GateResult(gate=self.name, verdict="approved", reason=f"wrote {len(written)} file(s)")


def _route_regex(method: str, path: str) -> re.Pattern[str]:
    return re.compile(r"@\w+\.(?:" + re.escape(method.lower()) + r"|api_route|route)\(\s*[\"']" + re.escape(path) + r"[\"']")


class EndpointPresentGate:
    """The endpoint the node reports (e.g. 'POST /login') must exist as a route in the code."""

    name = "endpoint_present"

    def __init__(self, result_key: str = "endpoint", glob: str = "app/**/*.py"):
        self.result_key = result_key
        self.glob = glob

    async def check(self, ctx: GateContext) -> GateResult:
        result = ctx.output.result or {}
        eps = [str(result.get(self.result_key, "")).strip()]
        eps += [str(e).strip() for e in (result.get("endpoints") or []) if str(e).strip() not in eps]
        sources = [(f, f.read_text(errors="ignore")) for f in ctx.workspace.glob(self.glob)]
        found: list[str] = []
        for ep in eps:
            parts = ep.split()
            if len(parts) != 2:
                return GateResult(gate=self.name, verdict="rejected", reason=f"endpoint must be 'METHOD /path', got {ep!r}")
            rx = _route_regex(*parts)
            hit = next((f for f, src in sources if rx.search(src)), None)
            if hit is None:
                return GateResult(gate=self.name, verdict="rejected",
                                  reason=f"reported endpoint {ep} is not defined in the code — add the route to app/main.py")
            found.append(f"{ep} in {hit.relative_to(ctx.workspace)}")
        return GateResult(gate=self.name, verdict="approved", reason="; ".join(found))


class UsesEndpointGate:
    """The consumer's code must reference the upstream endpoint path (e.g. in a fetch call)."""

    name = "uses_endpoint"

    def __init__(self, producer_agent: str, producer_dtype: str = "endpoint", glob: str = "static/**/*.html"):
        self.producer_agent = producer_agent
        self.producer_dtype = producer_dtype
        self.glob = glob

    async def check(self, ctx: GateContext) -> GateResult:
        ep = ctx.store.get_current(ctx.run_id, self.producer_agent, self.producer_dtype)
        if not ep:
            return GateResult(gate=self.name, verdict="rejected", reason=f"no :current {self.producer_agent}:{self.producer_dtype}")
        path = str(ep).split()[-1]
        for f in ctx.workspace.glob(self.glob):
            if path in f.read_text(errors="ignore"):
                return GateResult(gate=self.name, verdict="approved", reason=f"{path} referenced in {f.relative_to(ctx.workspace)}")
        return GateResult(gate=self.name, verdict="rejected",
                          reason=f"the UI never calls {path} — the fetch must target the backend endpoint from context")
