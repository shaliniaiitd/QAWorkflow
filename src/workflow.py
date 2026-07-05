"""File: src/workflow.py

LangGraph QA workflow (stateful graph), now with:
- Input guardrail (screen_input): blocks prompt-injection/SQLi-shaped
  user stories before they reach an LLM at all.
- Output guardrail + retry loop (validate_bdd): checks generated BDD has
  real Gherkin structure; loops back to regenerate (up to MAX_BDD_RETRIES)
  if not. This is the "intelligent workflow" branch point -- the graph's
  path now depends on what the LLM actually produced, not a fixed sequence.
- Observability (src/utils/observability.py): every LLM call is timed and
  logged locally (observability_log.jsonl), and traced to LangSmith if
  you've set the LANGCHAIN_TRACING_V2/LANGCHAIN_API_KEY env vars (safe
  no-op if you haven't).

Workflow in this file:
1. load_static_memory (reads project_memory.json)
2. screen_input (guardrail) -> blocked_report+save_report (if blocked)
                             -> retrieve_memory (if clean)
3. retrieve_memory (vector search for similar past stories)
4. analyze_story (creates acceptance criteria + edge cases)
5. write_dynamic_memory (curates what to remember this run)
6. generate_bdd (writes Gherkin)
7. validate_bdd (guardrail) -> generate_bdd (retry, up to MAX_BDD_RETRIES)
                             -> review_bdd (once valid or retries exhausted)
8. review_bdd (coverage/clarity check)
9. assemble_report (creates final markdown)
10. save_report (writes outputs/sample_report.md)

Prereqs:
- Ollama running on http://localhost:11434
- Model available locally (default: qwen2.5-coder:0.5b)
- Python deps: langgraph, langchain-ollama, langsmith (see requirements.txt)
- Optional, for LangSmith traces: LANGCHAIN_TRACING_V2=true,
  LANGCHAIN_API_KEY=<key>, LANGCHAIN_PROJECT=qa-workflow

How to run:
    python -m src.workflow
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()  # picks up .env at project root -- LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, etc.

from langchain_ollama import ChatOllama
from src.utils.vector_store import retrieve_similar_stories
from src.utils.vector_store import add_story_to_db
from src.utils.observability import traceable, timed_call, log_llm_call
from src.utils.guardrails import screen_user_story, check_bdd_output_schema


try:
    # LangGraph API changes occasionally; this import matches current common usage.
    from langgraph.graph import END, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: langgraph. Install with: pip install langgraph"
    ) from exc


MODEL_CONFIG = {
    "model_name": "qwen2.5-coder:0.5b",
    "temperature": 0.2,
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
}


PROJECT_DIR = Path(__file__).resolve().parents[1]
MEMORY_FILE = PROJECT_DIR / "project_memory.json"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUT_FILE = OUTPUTS_DIR /"sample_report.md"

PROMPT_FIELDS = {
    "analysis": ["static_memory", "retrieved_context","user_story"],
    "dynamic_memory": ["static_memory", "analysis"],
    "bdd": ["static_memory", "dynamic_memory", "user_story", "analysis"],
    "review": ["static_memory", "dynamic_memory", "bdd_cases"],
    "report": ["user_story", "static_memory", "retrieved_context", "dynamic_memory", "analysis", "bdd_cases", "review_notes"],
}

class QAState(TypedDict, total=False):
    # Inputs
    user_story: str

    # Memory
    static_memory: str
    dynamic_memory: str

    # Artifacts
    analysis: str
    bdd_cases: str
    review_notes: str
    final_report: str
    retrieved_context: str

    # Output
    output_path: str

    # Guardrails / intelligent-workflow control
    input_blocked: bool
    block_reason: str
    bdd_valid: bool
    retry_count: int


def create_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL_CONFIG["model_name"],
        temperature=MODEL_CONFIG["temperature"],
        base_url=MODEL_CONFIG["base_url"],
    )


@traceable(name="llm_text_call")
def _llm_text(llm: ChatOllama, prompt: str, node_name: str = "unknown") -> str:
    with timed_call() as t:
        msg = llm.invoke(prompt)
    text = msg.content if hasattr(msg, "content") else str(msg)
    usage = getattr(msg, "usage_metadata", None)
    log_llm_call(node_name, t.elapsed, len(prompt), len(text), usage)
    return text


_LLM: ChatOllama | None = None


def get_llm() -> ChatOllama:
    """Lazily construct (and cache) the LLM client.

    Deferred so importing this module never touches Ollama until a node
    actually runs a prompt -- important for mcp_server.py, which imports
    this module at startup before any tool has been invoked.
    """
    global _LLM
    if _LLM is None:
        _LLM = create_llm()
    return _LLM

#####################################
# Format and load STATIC MEMORY
#####################################
def _format_static_memory(memory: dict) -> str:
    common_risks = ", ".join(memory.get("common_risks", []))
    known_bug_patterns = ", ".join(memory.get("known_bug_patterns", []))
    return (
        f"Project name: {memory.get('project_name', 'Unknown')}\n"
        f"BDD style: {memory.get('bdd_style', 'Default')}\n"
        f"Common risks: {common_risks}\n"
        f"Known bug patterns: {known_bug_patterns}\n"
    )


def load_static_memory(state: QAState) -> QAState:
    if not MEMORY_FILE.exists():
        raise FileNotFoundError(
            f"Missing {MEMORY_FILE}. Create it (same one used in step 4/6/7)."
        )
    memory = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {"static_memory": _format_static_memory(memory)}

###############################
# Build and Format PROMPTS
##############################


def load_prompt(name: str) -> str:
    path = PROJECT_DIR / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")

def format_prompt(template_name: str, state: dict) -> str:
    # Load the actual template text from file
    template = load_prompt(template_name)
    fields = {field: state.get(field, "") for field in PROMPT_FIELDS[template_name]}
    return template.format(**fields)

####################################
# Define Nodes of the graph
###################################
@traceable(name="analyze_story")
def analyze_story(state: QAState) -> QAState:
    prompt = format_prompt("analysis", state)
    return {"analysis": _llm_text(get_llm(), prompt, node_name="analyze_story")}


@traceable(name="write_dynamic_memory")
def write_dynamic_memory(state: QAState) -> QAState:
    prompt = format_prompt("dynamic_memory", state)
    return {"dynamic_memory": _llm_text(get_llm(), prompt, node_name="write_dynamic_memory")}

@traceable(name="generate_bdd")
def generate_bdd(state: QAState) -> QAState:
    prompt = format_prompt("bdd", state)
    return {"bdd_cases": _llm_text(get_llm(), prompt, node_name="generate_bdd")}


@traceable(name="review_bdd")
def review_bdd(state: QAState) -> QAState:
    prompt = format_prompt("review" ,state)
    return {"review_notes": _llm_text(get_llm(), prompt, node_name="review_bdd")}


def assemble_report(state: QAState) -> QAState:
    report = format_prompt("report", state)
    return {"final_report": report}


def save_report(state: QAState) -> QAState:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(state.get("final_report", ""), encoding="utf-8")
    return {"output_path": str(OUTPUT_FILE)}

def retrieve_memory(state: QAState) -> QAState:
    query = state.get("user_story", "")
    similar = retrieve_similar_stories(query, top_k=3)
    return {"retrieved_context": "\n".join(similar)}


####################################
# Guardrails (input) + intelligent branching (output validation + retry)
###################################

def screen_input(state: QAState) -> QAState:
    """Input guardrail: block the user_story before it ever reaches an LLM
    if it looks like a prompt injection or SQL-injection-shaped string.
    See src/utils/guardrails.py for what's actually being checked.
    """
    result = screen_user_story(state.get("user_story", ""))
    return {"input_blocked": not result.passed, "block_reason": result.reason}


def route_after_screen(state: QAState) -> str:
    return "blocked_report" if state.get("input_blocked") else "retrieve_memory"


def blocked_report(state: QAState) -> QAState:
    """Short-circuit report for a user_story that failed the input guardrail.
    Skips retrieval, analysis, and generation entirely -- the whole point of
    an input guardrail is to stop before spending an LLM call on it."""
    reason = state.get("block_reason", "unspecified guardrail violation")
    report = (
        "# QA Report -- Blocked\n\n"
        "The submitted user story was blocked by an input guardrail before "
        "reaching the LLM.\n\n"
        f"Reason: {reason}\n"
    )
    return {"final_report": report}


MAX_BDD_RETRIES = 2


def validate_bdd(state: QAState) -> QAState:
    """Output guardrail: check the generated BDD actually looks like
    Gherkin before it's reviewed. This is the "intelligent workflow"
    branch point -- generate_bdd's output decides where the graph goes
    next, instead of always proceeding in a fixed sequence.
    """
    result = check_bdd_output_schema(state.get("bdd_cases", ""))
    retry_count = state.get("retry_count", 0)
    if not result.passed:
        retry_count += 1
    return {"bdd_valid": result.passed, "retry_count": retry_count}


def route_after_validate_bdd(state: QAState) -> str:
    if state.get("bdd_valid"):
        return "review_bdd"
    if state.get("retry_count", 0) >= MAX_BDD_RETRIES:
        # Stop retrying and proceed anyway rather than looping forever --
        # review_bdd will still flag it, just without another generation attempt.
        return "review_bdd"
    return "generate_bdd"


def build_graph(checkpointer=None):
    """Build the graph. Pass a checkpointer (e.g. MemorySaver()) to enable
    multi-turn memory: separate app.invoke() calls that share the same
    thread_id will resume from the SAME state instead of starting fresh --
    see run_multiturn() below for the pattern.
    """
    graph = StateGraph(QAState)
    graph.add_node("load_static_memory", load_static_memory)
    graph.add_node("screen_input", screen_input)
    graph.add_node("blocked_report", blocked_report)
    graph.add_node("analyze_story", analyze_story)
    graph.add_node("write_dynamic_memory", write_dynamic_memory)
    graph.add_node("generate_bdd", generate_bdd)
    graph.add_node("validate_bdd", validate_bdd)
    graph.add_node("review_bdd", review_bdd)
    graph.add_node("assemble_report", assemble_report)
    graph.add_node("save_report", save_report)
    graph.add_node("retrieve_memory", retrieve_memory)

    graph.set_entry_point("load_static_memory")
    graph.add_edge("load_static_memory", "screen_input")
    graph.add_conditional_edges(
        "screen_input",
        route_after_screen,
        {"blocked_report": "blocked_report", "retrieve_memory": "retrieve_memory"},
    )
    graph.add_edge("blocked_report", "save_report")
    graph.add_edge("retrieve_memory", "analyze_story")
    graph.add_edge("analyze_story", "write_dynamic_memory")
    graph.add_edge("write_dynamic_memory", "generate_bdd")
    graph.add_edge("generate_bdd", "validate_bdd")
    graph.add_conditional_edges(
        "validate_bdd",
        route_after_validate_bdd,
        {"generate_bdd": "generate_bdd", "review_bdd": "review_bdd"},
    )
    graph.add_edge("review_bdd", "assemble_report")
    graph.add_edge("assemble_report", "save_report")
    graph.add_edge("save_report", END)
    return graph.compile(checkpointer=checkpointer)


def run_multiturn(user_story: str, thread_id: str, checkpointer=None) -> QAState:
    """Run the workflow with multi-turn memory: two calls sharing the same
    thread_id continue from the SAME persisted state, rather than each
    starting from scratch.

    This is what "multiturn memory" means in an agent context: not the LLM
    literally remembering, but the graph's checkpointer saving state after
    every node and reloading it by thread_id on the next call. Useful for
    things like "the reviewer flagged an issue, let the user tweak the
    story and re-run just from generate_bdd onward" instead of re-running
    load_static_memory/retrieve_memory/analyze_story every time.

    Example:
        cp = MemorySaver()
        run_multiturn("As a user, I want to reset my password.", "session-1", cp)
        # ... later, same thread_id, same checkpointer instance ...
        run_multiturn("Also cover the case where the reset link expired.", "session-1", cp)
        # second call's state includes dynamic_memory/retrieved_context carried over from the first
    """
    checkpointer = checkpointer or MemorySaver()
    app = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"user_story": user_story}, config=config)


@traceable(name="run_full_qa_workflow")
def run_workflow(user_story: str) -> QAState:
    """Single entry point for one full workflow run, wrapped in ONE
    @traceable span. This is what makes node-level traces (analyze_story,
    generate_bdd, etc.) show up NESTED under one parent run in LangSmith,
    instead of each node's @traceable call appearing as its own
    independent top-level trace. mcp_server.py and app.py should call
    THIS function, not build_graph().invoke(...) directly, to get proper
    trace hierarchy.
    """
    app = build_graph()
    return app.invoke({"user_story": user_story})


def main() -> None:
    result = run_workflow(
        "As a user, I want to reset my password so I can regain access if I forget it."
    )
    print(f"Done. Wrote: {result.get('output_path')}")


if __name__ == "__main__":
    main()