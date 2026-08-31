# File: api.py (project root)
#
# FastAPI backend for the React frontend (frontend/). React runs in the
# browser and can't call Python functions directly -- it needs HTTP
# endpoints. Streamlit didn't need this because Streamlit's Python code
# runs server-side and renders its own UI; this is the actual structural
# difference between the two (see the comparison in chat).
#
# How to run:
#   pip install fastapi uvicorn
#   uvicorn api:app --reload --port 8000
#
# CORS is open to localhost:5173 (Vite's default dev port) below.

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.workflow import run_workflow
from src.utils.guardrails import screen_user_story, check_bdd_output_schema, check_with_llamaguard
from src.utils.observability import read_log
from rag_eval import evaluate as run_rag_evaluation

app1 = FastAPI(title="QA Workflow API")

app1.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserStoryRequest(BaseModel):
    user_story: str


class GuardrailCheckRequest(BaseModel):
    user_story: str = ""
    bdd_cases: str = ""
    use_llamaguard: bool = False


class RagEvalRequest(BaseModel):
    top_k: int = 3


@app1.post("/api/run-workflow")
def api_run_workflow(req: UserStoryRequest):
    result = run_workflow(req.user_story)
    return {
        "input_blocked": result.get("input_blocked", False),
        "block_reason": result.get("block_reason", ""),
        "analysis": result.get("analysis", ""),
        "bdd_cases": result.get("bdd_cases", ""),
        "review_notes": result.get("review_notes", ""),
        "bdd_valid": result.get("bdd_valid"),
        "retry_count": result.get("retry_count", 0),
        "final_report": result.get("final_report", ""),
        "output_path": result.get("output_path", ""),
    }


@app1.post("/api/check-guardrails")
def api_check_guardrails(req: GuardrailCheckRequest):
    results = {}
    if req.user_story:
        r = screen_user_story(req.user_story)
        results["user_story"] = {"passed": r.passed, "reason": r.reason}
    if req.bdd_cases:
        r = check_bdd_output_schema(req.bdd_cases)
        results["bdd_cases"] = {"passed": r.passed, "reason": r.reason}
    if req.use_llamaguard and req.user_story:
        r = check_with_llamaguard(req.user_story)
        results["llamaguard"] = {"passed": r.passed, "reason": r.reason}
    return results


@app1.post("/api/rag-eval")
def api_rag_eval(req: RagEvalRequest):
    results = run_rag_evaluation(top_k=req.top_k)
    hits = sum(1 for r in results if r.hit)
    return {
        "hits": hits,
        "total": len(results),
        "hit_rate": hits / len(results) if results else 0,
        "results": [
            {
                "query": r.query,
                "expected_section": r.expected_section,
                "hit": r.hit,
                "found_sections": r.found_sections,
            }
            for r in results
        ],
    }


@app1.get("/api/observability")
def api_observability():
    records = read_log()
    from collections import defaultdict
    import statistics

    by_node = defaultdict(list)
    for r in records:
        by_node[r["node"]].append(r["latency_s"])

    return {
        "total_calls": len(records),
        "by_node": [
            {
                "node": node,
                "calls": len(latencies),
                "mean_s": round(statistics.mean(latencies), 3),
                "max_s": round(max(latencies), 3),
            }
            for node, latencies in sorted(by_node.items())
        ],
    }


@app1.get("/api/health")
def health():
    return {"status": "ok"}