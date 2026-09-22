from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from society.gates import Gate
from society.providers import NodeOutput, Worker


class NodeSpec(BaseModel):
    id: str
    agent: str
    task: str
    depends_on: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list, description="KV refs 'agent:dtype' read from :current")
    outputs: list[str] = Field(default_factory=list, description="KV refs 'agent:dtype' this node publishes")
    tools: list[str] | None = Field(default=None, description="Override the agent's default tools")
    workspace: str | None = Field(default=None, description="Dir relative to run root; default = agent name")


class DagSpec(BaseModel):
    goal: str
    nodes: list[NodeSpec]

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


@dataclass
class AgentSpec:
    """Everything the runner needs to know about one agent, defined in code (Phase 5)."""

    name: str
    worker: Worker
    system_prompt: str
    tools: list[str] = field(default_factory=lambda: ["read_file", "write_file", "list_files", "run_command"])
    gates: list[Gate] = field(default_factory=list)
    outputs: dict[str, Callable[[NodeOutput], Any]] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    seed: Path | None = None
    description: str = ""
    prepare: Callable[[Path, dict[str, Any]], dict[str, Any] | None] | None = None


def parse_ref(ref: str) -> tuple[str, str]:
    agent, _, dtype = ref.partition(":")
    if not agent or not dtype:
        raise ValueError(f"bad KV ref {ref!r}; expected 'agent:dtype'")
    return agent, dtype
