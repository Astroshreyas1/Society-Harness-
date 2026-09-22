"""End-to-end run of the login DAG with scripted workers standing in for the models.

Proves the pipeline itself: seeding, per-agent workspaces, backend tests gate, schema
contract diff, merge at run root, merged tests gate, and the ship decision — all offline.
"""
import shutil
from pathlib import Path
from typing import Any

import pytest

from society.agents.registry import ROOT, build_agents
from society.gates import GateContext, GateResult, JevGate
from society.memory import KVStore, LineageLog
from society.providers import NodeInput, NodeOutput, Sandbox
from society.runner import DagRunner, load_dag

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


def _out(node: NodeInput, sb: Sandbox, result: dict[str, Any]) -> NodeOutput:
    return NodeOutput(node_id=node.node_id, agent=node.agent, provider="stub", model="stub",
                      result=result, files_written=sb.files_written, duration_ms=1)


class BackendStub:
    name = "stub"

    def __init__(self, wrong_first: bool = False):
        self.wrong_first = wrong_first
        self.calls = 0

    async def run(self, node: NodeInput) -> NodeOutput:
        self.calls += 1
        sb = Sandbox(node.workspace)
        sb.write_file("app/main.py", sb.read_file("app/main.py") + LOGIN_CODE)
        sb.write_file("tests/test_login.py", LOGIN_TEST)
        schema = RESPONSE_SCHEMA
        if self.wrong_first and self.calls == 1:
            sb.write_file("tests/test_login.py", LOGIN_TEST.replace("== 401", "== 403"))
        return _out(node, sb, {"endpoint": "POST /login", "request_schema": REQUEST_SCHEMA,
                               "response_schema": schema, "summary": "added /login"})


class FrontendStub:
    name = "stub"

    def __init__(self, drift: bool = False):
        self.drift = drift
        self.calls = 0
        self.seen: list[NodeInput] = []

    async def run(self, node: NodeInput) -> NodeOutput:
        self.calls += 1
        self.seen.append(node)
        sb = Sandbox(node.workspace)
        schema = node.context["backend:schema"]
        used = dict(schema)
        if self.drift and self.calls == 1:
            used = {"type": "object", "properties": {"authToken": {"type": "string"}, "user_id": {"type": "integer"}},
                    "required": ["authToken", "user_id"]}
        fields = list(used["properties"])
        html = sb.read_file("static/index.html").replace(
            "<h1>Todo</h1>",
            f'<h1>Todo</h1>\n<form id="login"><input id="u"><input id="p" type="password"><button>Login</button></form>'
            f'<!-- reads {", ".join(fields)} from POST /login -->')
        sb.write_file("static/index.html", html)
        return _out(node, sb, {"schema_used": used, "endpoint_called": node.context["backend:endpoint"],
                               "summary": "added login form"})


class IntegrationStub:
    name = "stub"

    async def run(self, node: NodeInput) -> NodeOutput:
        root = node.workspace
        for sub in ("app", "tests"):
            shutil.copytree(root / "backend" / sub, root / sub, dirs_exist_ok=True)
        shutil.copytree(root / "frontend" / "static", root / "static", dirs_exist_ok=True)
        shutil.copy(root / "backend" / "pytest.ini", root / "pytest.ini")
        sb = Sandbox(root)
        out = sb.run_command("python -m pytest -q tests")
        passed = "exit_code: 0" in out
        return _out(node, sb, {"merged_files": ["app/main.py", "static/index.html", "tests/test_login.py"],
                               "tests_passed": passed, "test_output": out[-500:], "summary": "merged"})


class ShipStub:
    name = "stub"

    async def run(self, node: NodeInput) -> NodeOutput:
        sb = Sandbox(node.workspace)
        tests_ok = "exit_code: 0" in sb.run_command("python -m pytest -q tests")
        has_login = '"/login"' in sb.read_file("app/main.py") and 'id="login"' in sb.read_file("static/index.html")
        checklist = [{"item": "tests", "ok": tests_ok}, {"item": "feature present", "ok": has_login}]
        return _out(node, sb, {"go": tests_ok and has_login, "checklist": checklist, "summary": "verified"})


class ApproveAll:
    name = "jev"

    async def check(self, ctx: GateContext) -> GateResult:
        return GateResult(gate=self.name, verdict="approved", confidence=0.99, reason="stubbed jev")


def _build(tmp_path: Path, backend: BackendStub, frontend: FrontendStub):
    workers = {"backend": backend, "frontend": frontend, "integration": IntegrationStub(), "ship": ShipStub()}
    agents = build_agents(workers)  # type: ignore[arg-type]
    for a in agents.values():
        a.gates = [ApproveAll() if isinstance(g, JevGate) else g for g in a.gates]
    kv, log = KVStore(), LineageLog(tmp_path / "l.jsonl")
    return DagRunner(agents, kv, log, tmp_path / "ws"), kv, log


async def test_login_dag_happy_path(tmp_path: Path):
    runner, kv, log = _build(tmp_path, BackendStub(), FrontendStub())
    res = await runner.run(load_dag(ROOT / "society/examples/login_dag.json"), run_id="r1")

    assert res.status == "completed", res.reason
    run = tmp_path / "ws/r1"
    assert (run / "backend/app/main.py").exists() and (run / "frontend/static/index.html").exists()
    assert '"/login"' in (run / "app/main.py").read_text(), "merged app must contain the endpoint"
    assert 'id="login"' in (run / "static/index.html").read_text(), "merged static must contain the form"
    assert kv.get_current("r1", "backend", "schema") == RESPONSE_SCHEMA
    assert kv.get_current("r1", "integration", "report")["tests_passed"] is True
    assert kv.get_current("r1", "ship", "decision")["go"] is True
    assert [r.verdict for r in log.read("r1")] == ["approved"] * 4


async def test_backend_test_failure_is_caught_and_retried(tmp_path: Path):
    backend = BackendStub(wrong_first=True)
    runner, kv, log = _build(tmp_path, backend, FrontendStub())
    res = await runner.run(load_dag(ROOT / "society/examples/login_dag.json"), run_id="r1")

    assert res.status == "completed", res.reason
    assert backend.calls == 2
    recs = [r for r in log.read("r1") if r.node_id == "backend"]
    assert recs[0].verdict == "rejected" and recs[0].gate == "backend_tests"
    assert recs[1].verdict == "approved"


async def test_frontend_schema_drift_is_caught_by_contract_gate(tmp_path: Path):
    frontend = FrontendStub(drift=True)
    runner, kv, log = _build(tmp_path, BackendStub(), frontend)
    res = await runner.run(load_dag(ROOT / "society/examples/login_dag.json"), run_id="r1")

    assert res.status == "completed", res.reason
    assert frontend.calls == 2
    recs = [r for r in log.read("r1") if r.node_id == "frontend"]
    assert recs[0].verdict == "rejected" and recs[0].gate == "schema_diff"
    assert "token" in recs[0].reason and "authToken" in recs[0].reason
    assert "[Retry" in frontend.seen[1].task and "schema mismatch" in frontend.seen[1].task
    assert "token, userId" in (tmp_path / "ws/r1/static/index.html").read_text(), "merged UI uses the contract"
