from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class NodeInput(BaseModel):
    node_id: str
    agent: str
    system_prompt: str
    task: str
    context: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    workspace: Path
    output_schema: dict[str, Any] | None = None
    attempt: int = 1
    run_id: str = ""                       # the DAG run this node belongs to (the harness keys its sessions by it)


class NodeOutput(BaseModel):
    node_id: str
    agent: str
    provider: str
    model: str
    text: str = ""
    result: dict[str, Any] | None = None
    files_written: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str | None = None
    error_kind: Literal["provider", "task"] | None = None
    fallback_from: str | None = None


class Worker(Protocol):
    name: str

    async def run(self, node: NodeInput) -> NodeOutput: ...
