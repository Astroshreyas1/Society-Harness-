"""The agents of the Society of LLMs (DAG-orchestrated software-team roles with gates, versioned KV handoff and a
retry loop), cloned into this repository so the harness (`society/harness`) can govern them: the roles, runner, gates
and memory are the original project's; the LLM providers are replaced by replay workers (`society/providers/replay.py`)
that do the roles' work by script with consumption drawn from the project's recorded runs."""
