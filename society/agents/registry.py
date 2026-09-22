"""The five roles of the Society of LLMs — prompts, output schemas, gates and KV extractors — as the original project
defines them, wired to replay workers instead of LLM providers. `make_workers(mode)` gives every role a `ReplayWorker`
on the provider the mode assigns it (mixed: Claude for backend / frontend, the local coder for integration / ship, as
the reference runs were made); `build_agents` is unchanged apart from the seed app's location."""
import sys
from pathlib import Path
from typing import Any

from society.gates import (CommandGate, DecisionConsistencyGate, EndpointPresentGate, FilesWrittenGate, JevGate,
                           OutputSchemaGate, SchemaDiffGate, UsesEndpointGate)
from society.providers import ReplayTable, ReplayWorker, Worker
from society.providers.replay import ToolHook
from society.runner import AgentSpec

from . import prompts

ROOT = Path(__file__).resolve().parents[2]
SEED_APP = ROOT / "society" / "examples" / "todo_app"
PYTEST = f"{sys.executable} -m pytest -q tests"

JSON_SCHEMA_OBJ: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "properties": {"type": "object"},
        "required": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["type", "properties"],
}

BACKEND_OUTPUT = {
    "type": "object",
    "properties": {
        "endpoint": {"type": "string", "description": "the primary endpoint the UI consumes, e.g. POST /login"},
        "endpoints": {"type": "array", "items": {"type": "string"}, "description": "every endpoint added, e.g. [\"POST /auth/request-otp\", \"POST /auth/verify-otp\"]"},
        "request_schema": JSON_SCHEMA_OBJ,
        "response_schema": JSON_SCHEMA_OBJ,
        "summary": {"type": "string"},
    },
    "required": ["endpoint", "request_schema", "response_schema", "summary"],
}

FRONTEND_OUTPUT = {
    "type": "object",
    "properties": {
        "schema_used": JSON_SCHEMA_OBJ,
        "endpoint_called": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["schema_used", "endpoint_called", "summary"],
}

INTEGRATION_OUTPUT = {
    "type": "object",
    "properties": {
        "merged_files": {"type": "array", "items": {"type": "string"}},
        "tests_passed": {"type": "boolean"},
        "test_output": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["merged_files", "tests_passed", "summary"],
}

SHIP_OUTPUT = {
    "type": "object",
    "properties": {
        "go": {"type": "boolean"},
        "checklist": {"type": "array", "items": {"type": "object", "properties": {
            "item": {"type": "string"}, "ok": {"type": "boolean"}, "note": {"type": "string"}},
            "required": ["item", "ok"]}},
        "summary": {"type": "string"},
    },
    "required": ["go", "checklist", "summary"],
}


def _r(o: Any, key: str) -> Any:
    return (o.result or {}).get(key)


def mechanical_merge(root: Path, context: dict[str, Any]) -> dict[str, Any]:
    """Deterministic merge: backend tree is the base, frontend's static/ overlays it."""
    import shutil

    merged: list[str] = []
    backend, frontend = root / "backend", root / "frontend"
    for sub in ("app", "tests", "static"):
        if (backend / sub).is_dir():
            shutil.copytree(backend / sub, root / sub, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
            merged += [f"{sub}/{p.name}" for p in (backend / sub).iterdir() if p.is_file()]
    for f in ("pytest.ini", "README.md"):
        if (backend / f).exists():
            shutil.copy(backend / f, root / f)
    if (frontend / "static").is_dir():
        shutil.copytree(frontend / "static", root / "static", dirs_exist_ok=True)
        merged += [f"static/{p.name}" for p in (frontend / "static").iterdir() if p.is_file()]
    return {"mechanical_merge": {
        "note": "The system already merged the trees at the run root before you started. "
                "Do not re-copy or rewrite files; verify the merge and run the tests.",
        "merged_files": sorted(set(merged)),
    }}


MODES: dict[str, dict[str, str]] = {
    "mixed": {"backend": "anthropic", "frontend": "anthropic", "integration": "ollama", "ship": "ollama"},
    "claude": {"backend": "anthropic", "frontend": "anthropic", "integration": "anthropic", "ship": "anthropic"},
    "local": {"backend": "ollama", "frontend": "ollama", "integration": "ollama", "ship": "ollama"},
}


def make_workers(mode: str = "mixed", table: ReplayTable | None = None, seed: int = 0, scale: float = 20.0,
                 tool_hook: ToolHook | None = None) -> dict[str, Worker]:
    """A replay worker per role on the provider `mode` assigns it. `tool_hook` lets the harness govern the shell tool
    calls the scripts make (pytest runs); `scale` divides recorded seconds into wall-clock seconds."""
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}; known: {sorted(MODES)}")
    table = table or ReplayTable()
    return {role: ReplayWorker(role, provider, table, seed=seed, scale=scale, tool_hook=tool_hook)
            for role, provider in MODES[mode].items()}


def build_agents(workers: dict[str, Worker], seed: Path = SEED_APP) -> dict[str, AgentSpec]:
    return {
        "backend": AgentSpec(
            name="backend", worker=workers["backend"], system_prompt=prompts.BACKEND,
            description=prompts.DESCRIPTIONS["backend"], seed=seed,
            output_schema=BACKEND_OUTPUT,
            outputs={
                "schema": lambda o: _r(o, "response_schema"),
                "request_schema": lambda o: _r(o, "request_schema"),
                "endpoint": lambda o: _r(o, "endpoint"),
            },
            gates=[OutputSchemaGate(), FilesWrittenGate(["app/", "tests/"]), EndpointPresentGate(),
                   CommandGate(PYTEST, name="backend_tests")],
        ),
        "frontend": AgentSpec(
            name="frontend", worker=workers["frontend"], system_prompt=prompts.FRONTEND,
            description=prompts.DESCRIPTIONS["frontend"], seed=seed,
            output_schema=FRONTEND_OUTPUT,
            outputs={"ui": lambda o: {"endpoint_called": _r(o, "endpoint_called"), "summary": _r(o, "summary"),
                                      "files": o.files_written}},
            gates=[OutputSchemaGate(), FilesWrittenGate(["static/"]), UsesEndpointGate("backend"),
                   SchemaDiffGate("backend", "schema")],
        ),
        "integration": AgentSpec(
            name="integration", worker=workers["integration"], system_prompt=prompts.INTEGRATION,
            description=prompts.DESCRIPTIONS["integration"], prepare=mechanical_merge,
            output_schema=INTEGRATION_OUTPUT,
            outputs={"report": lambda o: o.result},
            gates=[
                OutputSchemaGate(),
                CommandGate(PYTEST, name="merged_tests"),
                JevGate("The integration report must be consistent with a passing test run and must list the "
                        "merged files. Reject if the report claims tests passed but describes failures, or if "
                        "it says the merge was skipped."),
            ],
        ),
        "ship": AgentSpec(
            name="ship", worker=workers["ship"], system_prompt=prompts.SHIP,
            description=prompts.DESCRIPTIONS["ship"],
            tools=["read_file", "list_files", "run_command"],
            output_schema=SHIP_OUTPUT,
            outputs={"decision": lambda o: o.result},
            gates=[
                OutputSchemaGate(),
                DecisionConsistencyGate(),
                JevGate("Each checklist note must support its ok value: a note describing a failure, missing "
                        "feature, or failing tests cannot sit on an ok=true item. The summary must not contradict "
                        "the checklist. Reject if the notes and the verdicts disagree."),
            ],
        ),
    }
