"""One real call to TypeSafe's System One API through the controller's client, with a tiny observable state.

  export JEV_API_KEY=$(cat ~/.jev_api_key)
  python rung0/jev_smoke.py

Prints the typed answers with their probabilities and confidences, the model version, the input tokens and the
cost at the published price. Optional: JEV_API_BASE (default https://api.typesafe.ai; a LiteLLM proxy's
`<base>/typesafe` works), JEV_MODEL (default jev-latest).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentsim.jev import RemoteSystemOne, catalogue, questions_at  # noqa: E402

cat = catalogue(["bash", "read", "edit", "grep", "test", "search", "web", "api", "other"], ["plan", "explore", "edit", "verify", "final"])
model = RemoteSystemOne(cat)
state = {"dp": "chat_end", "sid": "demo", "recipe": "coding", "child": False, "request_idx": 0, "chats": 3, "tools": 4,
         "history": ["user", "chat", "tool:read", "tool:grep", "chat", "tool:edit", "chat"], "last_tool": "edit",
         "plan": "I've updated the parser. Now let me run the full test suite to make sure nothing else broke.",
         "content": [], "revealed": ["bash"], "revealed_content": ["tn:Bash in:8 pytest -q tests/ 2>&1 | tail -40"],
         "join_width": 0, "tokens_in": 41200, "age_s": 312.4, "occupancy": {"model.slots": 0.6, "sandbox.mem": 0.3, "sandbox.cpu": 0.4}}
qs = questions_at(cat, "chat_end", roles=("feature", "judge", "annotate"))
rec = model.decide(state, qs, 0.0)
print(f"model: {model.model}   questions: {len(qs)}   input tokens: {rec.usage_tokens}   cost: ${model.report()['cost_usd']}   latency: {model.report()['latency_p50_s']} s\n")
for q in qs:
    ans, p = rec.answers[q]
    conf = rec.confidence.get(q)
    extra = f"  score={rec.score(q):.2f}" if cat[q].type == "score" else (f"  P(yes)={rec.noul(q):.3f}" if cat[q].type == "noul" else "")
    print(f"{q:22s} {cat[q].type:6s} -> {ans:14s} p={p:.3f}" + (f"  confidence={conf:.3f}" if conf is not None else "") + extra)
print("\nfull distributions:")
print(json.dumps({q: [round(float(x), 3) for x in rec.dists[q]] for q in qs}, indent=1))
