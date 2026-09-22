from pathlib import Path

from society.providers import FallbackWorker, NodeInput, NodeOutput


class W:
    def __init__(self, name, model="m", error=None, kind=None):
        self.name, self.model, self.error, self.kind, self.calls = name, model, error, kind, 0

    async def run(self, node: NodeInput) -> NodeOutput:
        self.calls += 1
        return NodeOutput(node_id=node.node_id, agent=node.agent, provider=self.name, model=self.model,
                          result={"ok": True}, error=self.error, error_kind=self.kind)


def _node(tmp_path: Path, attempt: int = 1) -> NodeInput:
    return NodeInput(node_id="n", agent="a", system_prompt="s", task="t", tools=[], workspace=tmp_path, attempt=attempt)


async def test_provider_error_falls_back(tmp_path: Path):
    p, s = W("anthropic", error="RateLimitError: 429", kind="provider"), W("ollama", "devstral")
    out = await FallbackWorker(p, s).run(_node(tmp_path))
    assert out.provider == "ollama" and out.error is None
    assert out.fallback_from.startswith("anthropic:m — RateLimitError")
    assert p.calls == 1 and s.calls == 1


async def test_task_error_does_not_fall_back(tmp_path: Path):
    p, s = W("anthropic", error="final message was not valid JSON", kind=None), W("ollama")
    out = await FallbackWorker(p, s).run(_node(tmp_path))
    assert out.provider == "anthropic" and out.error and s.calls == 0, "gate rejections retry the same model"


async def test_switch_on_last_attempt(tmp_path: Path):
    p, s = W("ollama", "devstral"), W("ollama", "qwen3-coder")
    fw = FallbackWorker(p, s, switch_from_attempt=3)
    assert (await fw.run(_node(tmp_path, 1))).model == "devstral"
    assert (await fw.run(_node(tmp_path, 2))).model == "devstral"
    out = await fw.run(_node(tmp_path, 3))
    assert out.model == "qwen3-coder" and "attempt 3" in out.fallback_from
    assert p.calls == 2 and s.calls == 1


async def test_secondary_unavailable_returns_to_primary(tmp_path: Path):
    p, s = W("ollama", "devstral"), W("ollama", "qwen3-coder", error="NotFoundError: 404 model not found", kind="provider")
    out = await FallbackWorker(p, s, switch_from_attempt=3).run(_node(tmp_path, 3))
    assert out.model == "devstral" and out.error is None and "secondary unavailable" in out.fallback_from
