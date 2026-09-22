"""The agents without the LLMs: a `ReplayWorker` performs each role's work by script (the same file edits, test runs and
reports the e2e stub of the original project makes, so every deterministic gate runs for real) while its *resource
consumption* — wall-clock duration, input / output tokens, cost and whether the attempt comes back in a shape a gate
rejects — is drawn from `ReplayTable`: the recorded lineage of the original project's 23 runs (data/society/lineage.jsonl).

A draw is keyed by (seed, run, node, attempt), so every admission policy in the bench sees the same futures (common
random numbers). The draw is the sampled future of a node: only the worker (when the node runs) and the clairvoyant
policy (`ReplayTable.peek`) may read it — the harness's predictor must not, and the honesty test checks that.

Faults reproduce the rejections the recorded runs had, by the gate that caught them:
  no_error        the model's final message was not valid JSON (result None; files still written)
  output_schema   the structured result misses a required field
  files_written   the work was reported but no file was written
  backend_tests   the backend's own test asserts the wrong status code
  merged_tests    integration "fixes" the merge by adding a test that fails
  schema_diff     the frontend drifts from the published contract (never happened in the recordings; kept for tests)
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from .tools import Sandbox
from .types import NodeInput, NodeOutput

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = ROOT / "data" / "society" / "lineage.jsonl"

# USD per million tokens, first-party rates (the original project's anthropic_api.PRICING; local models cost nothing)
PRICING = {"anthropic": (5.0, 25.0), "ollama": (0.0, 0.0)}

REQUEST_SCHEMA = {"type": "object", "properties": {"username": {"type": "string"}, "password": {"type": "string"}},
                  "required": ["username", "password"]}
RESPONSE_SCHEMA = {"type": "object", "properties": {"token": {"type": "string"}, "userId": {"type": "integer"}},
                   "required": ["token", "userId"]}

LOGIN_CODE = '''

import secrets


class LoginIn(BaseModel):
    username: str
    password: str


@app.post("/login")
def login(body: LoginIn) -> dict:
    if body.username == "demo" and body.password == "demo123":
        return {"token": secrets.token_hex(16), "userId": 1}
    raise HTTPException(401, "invalid credentials")
'''

LOGIN_TEST = '''from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_login_ok():
    r = client.post("/login", json={"username": "demo", "password": "demo123"})
    assert r.status_code == 200 and set(r.json()) == {"token", "userId"} and r.json()["userId"] == 1


def test_login_bad():
    assert client.post("/login", json={"username": "demo", "password": "nope"}).status_code == 401
'''

BROKEN_TEST = '''def test_integration_probe():
    assert 1 == 2, "integration left a failing probe behind"
'''


class ProviderError(Exception):
    """The provider (or the platform's billing) refused the call: `kind` is rate_limit | overloaded | budget | timeout."""

    def __init__(self, kind: str, retry_after: float = 0.0, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind, self.retry_after = kind, retry_after


@dataclass(frozen=True)
class Draw:
    """One node attempt's sampled future."""
    duration_s: float
    tokens_in: int
    tokens_out: int
    fault: str | None           # the gate that will reject it, or None
    source_run: str


class ReplayTable:
    """Per (agent, provider) pools of recorded attempts. `split`: 'all' | 'even' | 'odd' selects the recorded runs the pool
    draws from, so a predictor can be warmed up on the other half (`records(split)`) without seeing the pool."""

    def __init__(self, path: Path = LINEAGE, split: str = "all"):
        if split not in ("all", "even", "odd"):
            raise ValueError("split must be all | even | odd")
        self.split = split
        recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        runs = sorted({r["run_id"] for r in recs})
        keep = {rid for i, rid in enumerate(runs) if split == "all" or (i % 2 == 0) == (split == "even")}
        self.pool: dict[tuple[str, str], list[dict]] = {}
        for r in recs:
            if r["run_id"] in keep and r.get("provider") and r["input_tokens"] > 0:   # real model attempts only
                self.pool.setdefault((r["agent"], r["provider"]), []).append(r)
        if not self.pool:
            raise ValueError(f"no usable records in {path} for split {split!r}")

    @staticmethod
    def records(path: Path = LINEAGE, split: str = "all") -> list[dict]:
        """The recorded attempts of the selected runs (what a deployment's own lineage log would show a predictor)."""
        return [r for pool in ReplayTable(path, split).pool.values() for r in pool]

    def providers(self) -> set[str]:
        return {prov for _, prov in self.pool}

    def _pool(self, agent: str, provider: str) -> list[dict]:
        pool = self.pool.get((agent, provider))
        if not pool:                                                    # a role never recorded on this provider: any provider's records
            pool = [r for (a, _), rs in self.pool.items() if a == agent for r in rs]
        if not pool:
            raise KeyError(f"no recorded attempts for agent {agent!r}")
        return pool

    @staticmethod
    def _rng(seed: int, run_id: str, node_id: str, attempt: int) -> np.random.Generator:
        return np.random.default_rng([seed, zlib.crc32(run_id.encode()), zlib.crc32(node_id.encode()), attempt])

    def draw(self, seed: int, run_id: str, node_id: str, attempt: int, agent: str, provider: str) -> Draw:
        pool = self._pool(agent, provider)
        rng = self._rng(seed, run_id, node_id, attempt)
        r = pool[int(rng.integers(len(pool)))]
        jitter = float(np.exp(rng.normal(0.0, 0.15)))                   # +-15 % lognormal around the recorded attempt
        fault = r["gate"] if r["verdict"] == "rejected" else None
        return Draw(duration_s=max(0.5, r["duration_ms"] / 1000.0 * jitter), tokens_in=int(r["input_tokens"] * jitter),
                    tokens_out=int(r["output_tokens"] * jitter), fault=fault, source_run=r["run_id"])

    allow_peek = False

    def peek(self, seed: int, run_id: str, node_id: str, attempt: int, agent: str, provider: str) -> Draw:
        """The clairvoyant policy's view of a future attempt. Anyone else calling this is cheating: the guard raises unless
        the bench has marked the table as the oracle's (the honesty test)."""
        if not self.allow_peek:
            raise RuntimeError("ReplayTable.peek: a future was read by a policy that is not the oracle")
        return self.draw(seed, run_id, node_id, attempt, agent, provider)


def cost_usd(provider: str, tokens_in: int, tokens_out: int) -> float:
    i, o = PRICING.get(provider, (5.0, 25.0))
    return (tokens_in * i + tokens_out * o) / 1_000_000


ToolHook = Callable[[str, str, Callable[[], str]], Awaitable[str]]     # (session id, tool kind, fn) -> output


class ReplayWorker:
    """One role's scripted behaviour with recorded consumption. `provider` names the endpoint the role would use
    (anthropic | ollama); `scale` divides recorded seconds into wall-clock seconds; `tool_hook`, when set, wraps every
    shell tool call the script makes (the harness takes a sandbox CPU unit around it)."""

    def __init__(self, agent: str, provider: str, table: ReplayTable, seed: int = 0, scale: float = 20.0,
                 tool_hook: ToolHook | None = None):
        if agent not in SCRIPTS:
            raise KeyError(f"no script for agent {agent!r} (have {sorted(SCRIPTS)})")
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.name = provider
        self.model = {"anthropic": "claude-opus-5", "ollama": "devstral"}.get(provider, provider)
        self.agent, self.provider, self.table, self.seed, self.scale = agent, provider, table, seed, scale
        self.tool_hook = tool_hook
        self.calls = 0

    def session(self, node: NodeInput) -> str:
        return f"{node.run_id}/{node.node_id}"

    async def _tool(self, node: NodeInput, kind: str, fn: Callable[[], str]) -> str:
        if self.tool_hook is None:
            return await asyncio.to_thread(fn)
        return await self.tool_hook(self.session(node), kind, fn)

    async def run(self, node: NodeInput) -> NodeOutput:
        self.calls += 1
        draw = self.table.draw(self.seed, node.run_id, node.node_id, node.attempt, self.agent, self.provider)
        t0 = time.monotonic()
        sb = Sandbox(node.workspace)
        out = NodeOutput(node_id=node.node_id, agent=node.agent, provider=self.name, model=self.model)
        result = await SCRIPTS[self.agent](self, node, sb, draw)
        out.result = result
        out.files_written = list(sb.files_written)
        if draw.fault == "no_error":
            out.result, out.error, out.error_kind = None, "final message was not valid JSON", "task"
        out.input_tokens, out.output_tokens = draw.tokens_in, draw.tokens_out
        out.cost_usd = cost_usd(self.provider, draw.tokens_in, draw.tokens_out)
        # the recorded duration covers the whole tool loop; sleep what the script's real work has not used yet
        remaining = draw.duration_s / self.scale - (time.monotonic() - t0)
        if remaining > 0:
            await asyncio.sleep(remaining)
        out.duration_ms = int((time.monotonic() - t0) * 1000 * self.scale)
        out.text = json.dumps(out.result) if out.result is not None else "I updated the files as requested."
        return out


# ---- the roles' scripts (the original project's e2e stub, with the recorded faults) --------------------------------
async def backend_script(w: ReplayWorker, node: NodeInput, sb: Sandbox, draw: Draw) -> dict[str, Any]:
    if draw.fault != "files_written":
        sb.write_file("app/main.py", sb.read_file("app/main.py").replace(LOGIN_CODE, "") + LOGIN_CODE)
        test = LOGIN_TEST.replace("== 401", "== 403") if draw.fault == "backend_tests" else LOGIN_TEST
        sb.write_file("tests/test_login.py", test)
        await w._tool(node, "run_command", lambda: sb.run_command("python -m pytest -q tests"))
    result = {"endpoint": "POST /login", "endpoints": ["POST /login"], "request_schema": REQUEST_SCHEMA,
              "response_schema": RESPONSE_SCHEMA, "summary": "added POST /login with a demo user and tests"}
    if draw.fault == "output_schema":
        del result["response_schema"]
    return result


async def frontend_script(w: ReplayWorker, node: NodeInput, sb: Sandbox, draw: Draw) -> dict[str, Any]:
    schema = node.context.get("backend:schema") or RESPONSE_SCHEMA
    endpoint = node.context.get("backend:endpoint") or "POST /login"
    used = dict(schema)
    if draw.fault == "schema_diff":
        used = {"type": "object", "properties": {"authToken": {"type": "string"}, "user_id": {"type": "integer"}},
                "required": ["authToken", "user_id"]}
    fields = ", ".join(used["properties"])
    path = endpoint.split()[-1]
    html = sb.read_file("static/index.html")
    if 'id="login"' not in html:
        html = html.replace("<h1>Todo</h1>", '<h1>Todo</h1>\n<form id="login"><input id="u"><input id="p" type="password">'
                            f'<button>Login</button></form>\n<!-- reads {fields} from {path} -->\n'
                            f'<script>async function login(u,p){{const r=await fetch("{path}",{{method:"POST",headers:{{"Content-Type":"application/json"}},'
                            'body:JSON.stringify({username:u,password:p})});return r.json();}</script>')
    sb.write_file("static/index.html", html)
    return {"schema_used": used, "endpoint_called": endpoint, "summary": "added a login form wired to the contract"}


async def integration_script(w: ReplayWorker, node: NodeInput, sb: Sandbox, draw: Draw) -> dict[str, Any]:
    root = node.workspace
    if not (root / "app").is_dir():                                     # the runner's prepare step merges; a bare workspace (tests) is merged here
        for sub in ("app", "tests"):
            shutil.copytree(root / "backend" / sub, root / sub, dirs_exist_ok=True)
        shutil.copytree(root / "frontend" / "static", root / "static", dirs_exist_ok=True)
        shutil.copy(root / "backend" / "pytest.ini", root / "pytest.ini")
    probe = root / "tests" / "test_integration_probe.py"
    if draw.fault == "merged_tests":
        sb.write_file("tests/test_integration_probe.py", BROKEN_TEST)
    elif probe.exists():                                                # the retry prompt names the failing probe: remove it
        probe.unlink()
        sb.files_written.append("tests/test_integration_probe.py")
    out = await w._tool(node, "run_command", lambda: sb.run_command("python -m pytest -q tests"))
    passed = "exit_code: 0" in out
    merged = (node.context.get("mechanical_merge") or {}).get("merged_files") or ["app/main.py", "static/index.html", "tests/test_login.py"]
    result = {"merged_files": merged, "tests_passed": passed, "test_output": out[-500:],
              "summary": "verified the merged tree; tests " + ("pass" if passed else "fail")}
    if draw.fault == "output_schema":
        del result["tests_passed"]
    return result


async def ship_script(w: ReplayWorker, node: NodeInput, sb: Sandbox, draw: Draw) -> dict[str, Any]:
    out = await w._tool(node, "run_command", lambda: sb.run_command("python -m pytest -q tests"))
    tests_ok = "exit_code: 0" in out
    has_login = '"/login"' in sb.read_file("app/main.py") and 'id="login"' in sb.read_file("static/index.html")
    checklist = [{"item": "tests", "ok": tests_ok, "note": "pytest " + ("green" if tests_ok else "red")},
                 {"item": "feature present", "ok": has_login, "note": "endpoint and form " + ("found" if has_login else "missing")},
                 {"item": "contract", "ok": True, "note": "the form posts to /login with username/password"},
                 {"item": "risk", "ok": True, "note": "demo credentials are hard-coded (demo app)"}]
    go = tests_ok and has_login
    if draw.fault == "decision_consistency":
        go = True
        checklist[0]["ok"] = False
    return {"go": go, "checklist": checklist, "summary": "ready to ship" if go else "blocked: see checklist"}


SCRIPTS: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
    "backend": backend_script, "frontend": frontend_script, "integration": integration_script, "ship": ship_script,
}
