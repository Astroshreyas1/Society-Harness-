import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from society.gates import validate_dag
from society.providers import NodeInput, Worker

from .spec import AgentSpec, DagSpec

PLANNER_SYSTEM = """You are the Planner for a multi-agent software team.
You do not write code. You produce a dependency graph (DAG) of tasks for the agents listed below.

Rules:
- Use ONLY the listed agents and ONLY the outputs each agent can publish.
- Every node's `inputs` must be outputs of nodes it (transitively) depends on.
- Independent work must NOT depend on each other so it can run in parallel.
- A node's `outputs` must be prefixed with its own agent name (e.g. "backend:schema").
- Tasks must be concrete, self-contained instructions. The agent will see nothing else
  except the KV values named in its `inputs`.
- Prefer fewer, well-scoped nodes. Do not add a node for an agent that has nothing to do.
- Output ONLY the JSON object. No prose."""


def dag_json_schema() -> dict[str, Any]:
    schema = DagSpec.model_json_schema()
    schema.pop("$defs", None)
    return {
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "agent": {"type": "string"},
                        "task": {"type": "string"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "inputs": {"type": "array", "items": {"type": "string"}},
                        "outputs": {"type": "array", "items": {"type": "string"}},
                        "workspace": {"type": ["string", "null"]},
                    },
                    "required": ["id", "agent", "task", "depends_on", "inputs", "outputs"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["goal", "nodes"],
        "additionalProperties": False,
    }


def agent_catalog(agents: dict[str, AgentSpec], descriptions: dict[str, str]) -> str:
    lines = []
    for name, spec in agents.items():
        outs = ", ".join(f"{name}:{d}" for d in spec.outputs) or "(none)"
        lines.append(f"- {name}: {descriptions.get(name, '')}\n  can publish: {outs}")
    return "\n".join(lines)


class Planner:
    def __init__(self, worker: Worker, agents: dict[str, AgentSpec], descriptions: dict[str, str],
                 workspace: Path):
        self.worker = worker
        self.agents = agents
        self.descriptions = descriptions
        self.workspace = workspace

    async def plan(self, goal: str) -> tuple[DagSpec | None, str]:
        """Returns (dag, reason). dag is None when the plan failed validation."""
        node = NodeInput(
            node_id="planner",
            agent="planner",
            system_prompt=PLANNER_SYSTEM + "\n\nAvailable agents:\n" + agent_catalog(self.agents, self.descriptions),
            task=f"Goal: {goal}\n\nProduce the DAG.",
            tools=[],
            workspace=self.workspace,
            output_schema=dag_json_schema(),
        )
        out = await self.worker.run(node)
        if out.error:
            return None, f"planner error: {out.error}"
        if out.result is None:
            return None, "planner returned no structured output"
        try:
            dag = DagSpec.model_validate(out.result)
        except ValidationError as e:
            return None, f"plan failed schema validation: {e}"
        res = validate_dag(dag.as_dict())
        if not res.approved:
            return None, f"plan rejected by dag gate: {res.reason}"
        return dag, "ok"


def load_dag(path: Path | str) -> DagSpec:
    return DagSpec.model_validate(json.loads(Path(path).read_text()))
