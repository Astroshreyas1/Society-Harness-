from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from society.memory import KVStore
from society.memory.lineage import Verdict
from society.providers import NodeInput, NodeOutput


class GateResult(BaseModel):
    gate: str
    verdict: Verdict
    confidence: float = 1.0
    reason: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.verdict == "approved"


class GateContext(BaseModel):
    run_id: str
    node: NodeInput
    output: NodeOutput
    attempt: int
    kv: Any
    workspace: Path
    prior: Any = None       # the verdicts of the gates that ran before this one on the attempt (the same list, appended as they run)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def store(self) -> KVStore:
        return self.kv


class Gate(Protocol):
    name: str

    async def check(self, ctx: GateContext) -> GateResult: ...
