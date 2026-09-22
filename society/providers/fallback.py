from .types import NodeInput, NodeOutput, Worker


class FallbackWorker:
    """Run `primary`; hand the node to `secondary` when the provider itself fails
    (auth, rate limit, network) or, optionally, from a given attempt number onward
    so a stuck model gets a different one for its last try."""

    def __init__(self, primary: Worker, secondary: Worker, switch_from_attempt: int | None = None):
        self.primary = primary
        self.secondary = secondary
        self.switch_from_attempt = switch_from_attempt
        self.name = primary.name
        self.model = getattr(primary, "model", "")

    async def run(self, node: NodeInput) -> NodeOutput:
        if self.switch_from_attempt and node.attempt >= self.switch_from_attempt:
            out = await self.secondary.run(node)
            if out.error_kind == "provider":
                # The fallback itself is unavailable (e.g. model not pulled): stay on the primary.
                reason = out.error
                out = await self.primary.run(node)
                out.fallback_from = f"secondary unavailable — {reason}"
                return out
            out.fallback_from = f"{self.primary.name}:{getattr(self.primary, 'model', '')} (attempt {node.attempt})"
            return out
        out = await self.primary.run(node)
        if out.error_kind == "provider":
            reason = out.error
            out = await self.secondary.run(node)
            out.fallback_from = f"{self.primary.name}:{getattr(self.primary, 'model', '')} — {reason}"
        return out
