"""The System One gate of the Society of LLMs, re-pointed at this repository's one Jev client (agentsim/jev.py).

The original project asked Jev (or a local Qwen stand-in through Ollama) one yes/no question over the evidence of a
node attempt: does the agent's output satisfy the criteria, with the structured result and the files written as ground
truth and the agent's message as a claim. This clone keeps that question and offers two backends:

  remote   TypeSafe's System One through `agentsim.jev.RemoteSystemOne` (JEV_API_KEY, JEV_API_BASE, ...): the noul
           question `approve` plus the judge-style questions of the society catalogue in one call; the answer's
           probability is the confidence, and below `abstain_below` the gate abstains (the executor escalates).
  rule     the deterministic stand-in used when no model is reachable: the same evidence, the same criteria, checked
           mechanically — the report must agree with the gate evidence recorded before it (a claimed passing test run
           against a failed `merged_tests`, a checklist note that contradicts its ok flag, a merge reported as skipped).
           It is a rule, not a model, and says so in every verdict (`gate = "jev_rule"`), so a lineage never mistakes
           one for the other.

`JEV_GATE_BACKEND` (remote | rule) selects the default; `rule` is used when it is unset or the key is missing.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

from .types import GateContext, GateResult

FAILURE_WORDS = re.compile(r"\b(fail(?:ed|ing|s)?|error|broken|missing|not (?:found|present|implemented)|crash|cannot|skipped)\b", re.I)


def society_catalogue() -> dict:
    """The society's questions for System One, in the harness's Question form (agentsim.jev)."""
    from agentsim.jev import Question

    Q = [
        Question("approve", "noul",
                 "Does the agent's output satisfy `criteria`? `files_written_ground_truth` and `structured_result_ground_truth` were "
                 "recorded by the system; `agent_final_message_claim` is only a claim and `gate_evidence` lists the verdicts of "
                 "the deterministic checks that already ran.",
                 ("no", "yes"), "yes when the ground truth supports every claim the output makes and no criterion is violated",
                 ("gate",), "verdict", "judge"),
        Question("claim_contradicts_evidence", "noul",
                 "Does `agent_final_message_claim` or `structured_result_ground_truth` state something that `gate_evidence` or "
                 "`files_written_ground_truth` contradicts (a passing test run that failed, a file that was not written)?",
                 ("no", "yes"), "yes when a stated fact is contradicted by the recorded evidence", ("gate",), "verdict", "verify"),
        Question("needs_human", "noul",
                 "Would a careful reviewer want a human to look at this attempt before it is used downstream?",
                 ("no", "yes"), "yes when the evidence is insufficient to decide either way", ("gate",), "verdict", "annotate"),
    ]
    return {q.name: q for q in Q}


class JevGate:
    """System One gate: `remote` through the repository's client, else the deterministic `rule` stand-in."""

    name = "jev"

    def __init__(self, criteria: str, backend: str | None = None, abstain_below: float = 0.6, timeout_s: float = 15.0):
        self.criteria = criteria
        self.abstain_below = abstain_below
        want = (backend or os.environ.get("JEV_GATE_BACKEND") or "rule").lower()
        if want not in ("remote", "rule"):
            raise ValueError("JevGate backend must be remote | rule")
        self.backend = "remote" if want == "remote" and os.environ.get("JEV_API_KEY") else "rule"
        self.name = "jev" if self.backend == "remote" else "jev_rule"
        self._model = None
        self.timeout_s = timeout_s
        self.asked, self.abstained = 0, 0

    # ---- evidence ----------------------------------------------------------------------------------
    def _state(self, ctx: GateContext) -> dict[str, Any]:
        out = ctx.output
        prior = ctx.prior or []
        state: dict[str, Any] = {
            "dp": "gate", "criteria": self.criteria, "agent": ctx.node.agent, "attempt": ctx.attempt,
            "files_written_ground_truth": list(out.files_written),
            "structured_result_ground_truth": out.result,
            "agent_final_message_claim": (out.text or "")[:2000],
            "gate_evidence": [{"gate": g.gate, "verdict": g.verdict, "reason": g.reason[:300]} for g in prior],
            "task_background": ctx.node.task[:800],
        }
        if out.error:
            state["worker_error"] = out.error
        return state

    # ---- backends ------------------------------------------------------------------------------------
    async def _remote(self, ctx: GateContext) -> GateResult:
        import asyncio

        from agentsim.jev import RemoteSystemOne

        if self._model is None:
            self._model = RemoteSystemOne(society_catalogue(), timeout_s=self.timeout_s)
        state = self._state(ctx)
        t0 = time.monotonic()
        rec = await asyncio.to_thread(self._model.decide, state, ("approve", "claim_contradicts_evidence", "needs_human"), 0.0)
        ms = int((time.monotonic() - t0) * 1000)
        p_yes = rec.noul("approve") or 0.0
        conf = max(p_yes, 1.0 - p_yes)
        details = {"ms": ms, "model": self._model.model, "p_approve": round(p_yes, 3),
                   "p_contradiction": round(rec.noul("claim_contradicts_evidence") or 0.0, 3),
                   "p_needs_human": round(rec.noul("needs_human") or 0.0, 3)}
        reason = f"jev {self._model.model}: approve={p_yes >= 0.5} (p={conf:.2f})"
        if conf < self.abstain_below:
            self.abstained += 1
            return GateResult(gate=self.name, verdict="abstained", confidence=conf, reason=f"low confidence ({conf:.2f}): {reason}", details=details)
        return GateResult(gate=self.name, verdict="approved" if p_yes >= 0.5 else "rejected", confidence=conf, reason=reason, details=details)

    def _rule(self, ctx: GateContext) -> GateResult:
        out = ctx.output
        r = out.result or {}
        prior = {g.gate: g for g in (ctx.prior or [])}
        problems: list[str] = []
        if out.error:
            problems.append(f"worker error: {out.error}")
        # integration: the report must agree with the merged test run and list the merge
        if "tests_passed" in r:
            tests = prior.get("merged_tests")
            if tests is not None and bool(r["tests_passed"]) != tests.approved:
                problems.append(f"report says tests_passed={r['tests_passed']} but merged_tests {tests.verdict}")
            if r.get("tests_passed") and FAILURE_WORDS.search(str(r.get("summary", ""))) and not re.search(r"\b(no|zero|without)\b[^.]*\bfail", str(r.get("summary", "")), re.I):
                problems.append("summary describes failures while tests_passed is true")
            if not r.get("merged_files"):
                problems.append("merged_files is empty")
            if re.search(r"merge (?:was )?skipped", str(r.get("summary", "")), re.I):
                problems.append("summary says the merge was skipped")
        # ship: every note must support its ok flag; the summary must not contradict the decision
        if "checklist" in r:
            for item in r.get("checklist") or []:
                note = str(item.get("note", ""))
                if item.get("ok") and FAILURE_WORDS.search(note) and not re.search(r"\b(no|zero|without|green)\b", note, re.I):
                    problems.append(f"checklist item {item.get('item')!r} is ok=true but its note reads as a failure: {note!r}")
            summary = str(r.get("summary", ""))
            if r.get("go") is True and re.search(r"\b(blocked|not ready|do not ship|no-go)\b", summary, re.I):
                problems.append("summary says not to ship while go=true")
            if r.get("go") is False and re.search(r"\bready to ship\b", summary, re.I):
                problems.append("summary says ready to ship while go=false")
        self.asked += 1
        if problems:
            return GateResult(gate=self.name, verdict="rejected", confidence=1.0, reason="; ".join(problems), details={"rule": True})
        return GateResult(gate=self.name, verdict="approved", confidence=1.0, reason="report consistent with the recorded evidence (rule stand-in, not a model)",
                          details={"rule": True})

    async def check(self, ctx: GateContext) -> GateResult:
        if self.backend == "remote":
            try:
                self.asked += 1
                return await self._remote(ctx)
            except Exception as e:  # noqa: BLE001 — the model is unreachable: the rule decides, and says so
                res = self._rule(ctx)
                res.reason += f" [jev api unavailable ({type(e).__name__}); rule stand-in used]"
                return res
        return self._rule(ctx)
