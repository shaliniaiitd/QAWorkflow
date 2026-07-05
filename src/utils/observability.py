"""File: src/utils/observability.py

Observability for LLM calls: traces, latency, token usage -- the LLM
equivalent of what you already instrument for API calls in SDET work.
Traces = logs. Evals = assertions. Latency baselines = performance testing.
Same discipline, different subject.

Two layers, independent of each other:

1. LangSmith tracing (@traceable) -- rich, hierarchical traces viewable in
   the LangSmith UI (which node called which LLM with what prompt, nested
   under the parent workflow run). Opt-in, zero-config-safe: with no env
   vars set, @traceable is a transparent no-op -- your code runs exactly
   as before, nothing is uploaded, nothing breaks. To actually see traces:

   from os import dotenv
   load_dotenv()  # if you haven't already loaded .env

       export LANGCHAIN_TRACING_V2=true
       export LANGCHAIN_API_KEY= os.getenv("LANGCHAIN_API_KEY")  # or your own key
       export LANGCHAIN_PROJECT=qa-workflow          # optional, else "default"

   Get a key at https://smith.langchain.com (free tier is enough for this).

2. Local JSONL log (log_llm_call) -- works with NO account and NO network,
   because Ollama is local and free, so "cost tracking" in the traditional
   sense doesn't apply here. What we log instead: latency per call, prompt/
   response size, and token counts when the model reports them. This is
   what scripts/observability_report.py reads to compute latency baselines
   -- the same idea as tracking p95 API response time, just for LLM calls.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*_args: Any, **_kwargs: Any):
        """Fallback if langsmith isn't installed: decorator becomes a no-op."""
        def decorator(fn):
            return fn
        return decorator

PROJECT_DIR = Path(__file__).resolve().parents[2]
LOG_FILE = PROJECT_DIR / "observability_log.jsonl"

def log_llm_call(
    node_name: str,
    latency_s: float,
    prompt_chars: int,
    response_chars: int,
    usage: Any | None = None,
) -> dict[str, Any]:
    """Append one LLM call's stats to observability_log.jsonl.

    Args:
        node_name: Which workflow step made the call, e.g. "analyze_story".
        latency_s: Wall-clock seconds for the call.
        prompt_chars: len(prompt) -- a free, always-available size proxy.
        response_chars: len(response text).
        usage: The response's usage_metadata if the model reports one
            (ChatOllama/ChatOpenAI expose input_tokens/output_tokens/
            total_tokens on many models). None if unavailable.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node": node_name,
        "latency_s": round(latency_s, 3),
        "prompt_chars": prompt_chars,
        "response_chars": response_chars,
        "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
        "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_log() -> list[dict[str, Any]]:
    """Read every logged call. Used by the report script and rag_eval's
    history tracking."""
    if not LOG_FILE.exists():
        return []
    records = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class timed_call:
    """Context manager measuring wall-clock latency around an LLM call.

    Usage:
        with timed_call() as t:
            msg = llm.invoke(prompt)
        log_llm_call("analyze_story", t.elapsed, len(prompt), len(msg.content))
    """

    def __enter__(self) -> "timed_call":
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed = time.perf_counter() - self._start