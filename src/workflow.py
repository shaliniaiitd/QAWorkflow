"""File: workflows/08_langgraph_qa_workflow.py

Step 8: LangGraph QA workflow (stateful graph).

Learning goal:
- Represent our QA workflow as an explicit graph with shared state.
- See how "workflow steps" become "graph nodes".
- Keep memory ownership explicit (a memory-writer node updates dynamic memory).

Focus of this file:
- orchestration with LangGraph (graph + state), not tool calling
- using static memory + dynamic memory in downstream steps
- saving outputs under the project-level outputs/ folder

Workflow in this file:
1. load_static_memory (reads project_memory.json)
2. analyze_story (creates acceptance criteria + edge cases)
3. write_dynamic_memory (curates what to remember this run)
4. generate_bdd (writes Gherkin)
5. review_bdd (coverage/clarity check)
6. assemble_report (creates final markdown)
7. save_report (writes outputs/sample_report.md)

Prereqs:
- Ollama running on http://localhost:11434
- Model available locally (default: qwen3:latest)
- Python deps: langgraph, langchain-ollama

How to run:
    python workflow.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from langchain_ollama import ChatOllama
from src.utils.vector_store import retrieve_similar_stories
from src.utils.vector_store import add_story_to_db


try:
    # LangGraph API changes occasionally; this import matches current common usage.
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: langgraph. Install with: pip install langgraph"
    ) from exc


MODEL_CONFIG = {
    "model_name": "qwen2.5-coder:0.5b",
    "temperature": 0.2,
    "base_url": "http://localhost:11434",
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


def create_llm() -> ChatOllama:
    return ChatOllama(
        model=MODEL_CONFIG["model_name"],
        temperature=MODEL_CONFIG["temperature"],
        base_url=MODEL_CONFIG["base_url"],
    )


def _llm_text(llm: ChatOllama, prompt: str) -> str:
    msg = llm.invoke(prompt)
    return msg.content if hasattr(msg, "content") else str(msg)


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
    path = Path(__file__).resolve().parents[0] / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")

def format_prompt(template_name: str, state: dict) -> str:
    # Load the actual template text from file
    template = load_prompt(template_name)
    fields = {field: state.get(field, "") for field in PROMPT_FIELDS[template_name]}
    return template.format(**fields)

####################################
# Define Nodes of the graph
###################################
def analyze_story(state: QAState) -> QAState:
    prompt = format_prompt("analysis", state)
    return {"analysis": _llm_text(get_llm(), prompt)}


def write_dynamic_memory(state: QAState) -> QAState:
    prompt = format_prompt("dynamic_memory", state)
    return {"dynamic_memory": _llm_text(get_llm(), prompt)}

def generate_bdd(state: QAState) -> QAState:
    prompt = format_prompt("bdd", state)
    return {"bdd_cases": _llm_text(get_llm(), prompt)}


def review_bdd(state: QAState) -> QAState:
    prompt = format_prompt("review" ,state)
    return {"review_notes": _llm_text(get_llm(), prompt)}


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


def build_graph():
    graph = StateGraph(QAState)
    graph.add_node("load_static_memory", load_static_memory)
    graph.add_node("analyze_story", analyze_story)
    graph.add_node("write_dynamic_memory", write_dynamic_memory)
    graph.add_node("generate_bdd", generate_bdd)
    graph.add_node("review_bdd", review_bdd)
    graph.add_node("assemble_report", assemble_report)
    graph.add_node("save_report", save_report)
    graph.add_node("retrieve_memory", retrieve_memory)


    graph.set_entry_point("load_static_memory")
    graph.add_edge("load_static_memory", "retrieve_memory")
    graph.add_edge("retrieve_memory", "analyze_story")
    graph.add_edge("analyze_story", "write_dynamic_memory")
    graph.add_edge("write_dynamic_memory", "generate_bdd")
    graph.add_edge("generate_bdd", "review_bdd")
    graph.add_edge("review_bdd", "assemble_report")
    graph.add_edge("assemble_report", "save_report")
    graph.add_edge("save_report", END)
    return graph.compile()


def main() -> None:
    app = build_graph()
    initial_state: QAState = {
        "user_story": "As a user, I want to reset my password so I can regain access if I forget it."
    }

    # If you want step-by-step visibility later, you can switch to:
    # for l_state = app.invoke(initial_state):
    #     print(f"Done. Wrote: {final_state.get('output_path')}")

    final_path = None
    for event in app.stream(initial_state):
        if "save_report" in event:
            final_path = event["save_report"].get("output_path")
            break

    print(f"Done. Wrote: {final_path}")


if __name__ == "__main__":
    main()

