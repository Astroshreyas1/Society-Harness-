import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable


class SandboxViolation(Exception):
    pass


class Sandbox:
    """File and shell tools hard-locked to one workspace directory."""

    def __init__(self, workspace: Path):
        self.root = workspace.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.files_written: list[str] = []

    def _resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise SandboxViolation(f"path escapes workspace: {rel}")
        return p

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file at a path relative to the workspace root."""
        try:
            return self._resolve(path).read_text()
        except FileNotFoundError:
            return f"ERROR: file not found: {path}"
        except SandboxViolation as e:
            return f"ERROR: {e}"

    def write_file(self, path: str, content: str) -> str:
        """Write content to a file at a path relative to the workspace root. Creates parent dirs."""
        try:
            p = self._resolve(path)
        except SandboxViolation as e:
            return f"ERROR: {e}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        rel = str(p.relative_to(self.root))
        if rel not in self.files_written:
            self.files_written.append(rel)
        return f"wrote {len(content)} bytes to {rel}"

    def list_files(self) -> str:
        """List all files in the workspace, relative to its root."""
        return "\n".join(
            str(p.relative_to(self.root))
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts
        ) or "(empty)"

    def run_command(self, command: str) -> str:
        """Run a shell command inside the workspace and return stdout, stderr, and the exit code."""
        import subprocess

        try:
            r = subprocess.run(
                command, shell=True, cwd=self.root, capture_output=True, text=True, timeout=120,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return "ERROR: command timed out after 120s"
        return f"exit_code: {r.returncode}\nstdout:\n{r.stdout[-6000:]}\nstderr:\n{r.stderr[-6000:]}"

    def callables(self, names: list[str]) -> list[Callable[..., str]]:
        return [getattr(self, n) for n in names]

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        return getattr(self, name)(**args)


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file at a path relative to the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at a path relative to the workspace root. Creates parent dirs.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the workspace, relative to its root.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "run_command": {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command inside the workspace and return stdout, stderr, and the exit code.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
}


def openai_tool_schemas(names: list[str]) -> list[dict[str, Any]]:
    return [TOOL_SCHEMAS[n] for n in names]


def format_context(context: dict[str, Any]) -> str:
    if not context:
        return ""
    parts = ["## Context from upstream agents (read-only)"]
    for k, v in context.items():
        body = v if isinstance(v, str) else json.dumps(v, indent=2)
        parts.append(f"### {k}\n{body}")
    return "\n\n".join(parts)


async def to_thread(fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
    return await asyncio.to_thread(fn, *a, **kw)
