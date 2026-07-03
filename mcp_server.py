"""File: mcp_server.py (project root, alongside seed_vector_db.py, check_retrieval.py, rag_eval.py)

MCP server exposing the QA workflow's LangGraph nodes as individual tools,
plus the RAG pipeline (user story generation, seeding, retrieval checkup,
and eval) built around Shalini's portfolio content as the source material.

Why this design:
- Each node function in src/workflow.py already takes a partial `QAState`
  dict and returns a partial dict (e.g. `analyze_story(state) ->
  {"analysis": ...}`). That's exactly the shape an MCP tool needs:
  explicit inputs, one clear output.
- Rather than duplicating logic, this file *imports* the existing modules
  and thinly wraps them, so the graph (batch path), the CLI scripts
  (seed_vector_db.py, check_retrieval.py, rag_eval.py), and the MCP tools
  (interactive/agentic calls from Claude Desktop or Claude Code) all share
  one implementation instead of drifting apart.

How to run:
    python mcp_server.py                       # stdio transport

Prereqs:
- Ollama running on http://localhost:11434 with both models pulled:
    ollama pull qwen2.5-coder:0.5b     # chat model, see src/workflow.py MODEL_CONFIG
    ollama pull nomic-embed-text       # embedding model, see src/utils/vector_store.py
- `project_memory.json` present at the project root
- `prompts/analysis.txt`, `prompts/bdd.txt`, `prompts/review.txt`,
  `prompts/dynamic_memory.txt`, `prompts/report.txt` present
- Python deps: see requirements.txt
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

import src.workflow as wf
import seed_vector_db as seed
from rag_eval import evaluate as run_rag_evaluation
from scripts.generate_user_stories import (
    generate_stories_for_section,
    load_portfolio_sections,
    write_story_file,
)
from src.utils.vector_store import collection_stats, retrieve_similar_chunks

mcp = FastMCP(
    "qa-workflow",
    instructions=(
        "Tools for turning a user story into reviewed BDD/Gherkin test cases, "
        "backed by a project-specific static memory file and a vector store "
        "of similar past stories. Typical order: retrieve_similar_stories -> "
        "analyze_story -> generate_bdd -> review_bdd. Use run_full_qa_workflow "
        "to do all of that plus write the markdown report in one call."
    ),
)


@mcp.tool()
def retrieve_similar_stories(user_story: str, top_k: int = 3) -> str:
    """Retrieve the top-k most similar past user stories from the vector store.

    Use this first, before analyze_story, to ground the analysis in
    precedent from stories already stored in the project's vector DB.
    """
    result = wf.retrieve_memory({"user_story": user_story})
    return result["retrieved_context"]


@mcp.tool()
def load_static_memory() -> str:
    """Load the project's static memory (project name, BDD style, known
    risks and bug patterns) from project_memory.json.

    Call this once per session and pass its output as `static_memory` to
    the other tools, so their output stays consistent with project
    conventions.
    """
    return wf.load_static_memory({})["static_memory"]


@mcp.tool()
def analyze_story(
    user_story: str,
    static_memory: str = "",
    retrieved_context: str = "",
) -> str:
    """Analyze a user story into acceptance criteria and edge cases.

    Args:
        user_story: The raw user story text, e.g. "As a user, I want to
            reset my password so I can regain access if I forget it."
        static_memory: Output of load_static_memory (project conventions).
        retrieved_context: Output of retrieve_similar_stories, for context
            from similar past stories.
    """
    state = {
        "user_story": user_story,
        "static_memory": static_memory,
        "retrieved_context": retrieved_context,
    }
    return wf.analyze_story(state)["analysis"]


@mcp.tool()
def write_dynamic_memory(static_memory: str, analysis: str) -> str:
    """Curate what should be remembered from this run for future runs.

    Args:
        static_memory: Output of load_static_memory.
        analysis: Output of analyze_story.
    """
    state = {"static_memory": static_memory, "analysis": analysis}
    return wf.write_dynamic_memory(state)["dynamic_memory"]


@mcp.tool()
def generate_bdd(
    user_story: str,
    analysis: str,
    static_memory: str = "",
    dynamic_memory: str = "",
) -> str:
    """Generate Gherkin BDD test cases from a user story and its analysis.

    Args:
        user_story: The raw user story text.
        analysis: Output of analyze_story.
        static_memory: Output of load_static_memory.
        dynamic_memory: Output of write_dynamic_memory (optional but
            recommended for consistency across a session).
    """
    state = {
        "user_story": user_story,
        "analysis": analysis,
        "static_memory": static_memory,
        "dynamic_memory": dynamic_memory,
    }
    return wf.generate_bdd(state)["bdd_cases"]


@mcp.tool()
def review_bdd(
    bdd_cases: str,
    static_memory: str = "",
    dynamic_memory: str = "",
) -> str:
    """Review generated BDD cases for coverage gaps and clarity issues.

    Args:
        bdd_cases: Output of generate_bdd.
        static_memory: Output of load_static_memory.
        dynamic_memory: Output of write_dynamic_memory.
    """
    state = {
        "bdd_cases": bdd_cases,
        "static_memory": static_memory,
        "dynamic_memory": dynamic_memory,
    }
    return wf.review_bdd(state)["review_notes"]


@mcp.tool()
def add_story_to_vector_store(story_id: str, story_text: str, section: str = "") -> str:
    """Add a single story/chunk directly to the vector store.

    For bulk (re)indexing everything under user_stories/, use
    reindex_vector_db instead -- this is for adding one-off entries.

    Args:
        story_id: A unique identifier, e.g. "story4" or "projects_3::chunk0".
        story_text: The text to embed and store.
        section: Optional category, e.g. "projects", "skills" -- enables
            filtered retrieval later via search_portfolio(section=...).
    """
    wf.add_story_to_db(story_id, story_text, metadata={"section": section} if section else None)
    return f"Stored '{story_id}' in the vector store."


@mcp.tool()
def generate_user_stories(section: str = "", count: int = 2) -> str:
    """Generate new user stories grounded in the portfolio content
    (data/portfolio_content.json) and write them to user_stories/<section>/.

    This does NOT seed the vector store -- call reindex_vector_db
    afterwards to pick up the new files.

    Args:
        section: One portfolio section to generate for, e.g. "projects".
            Leave empty to generate for every section.
        count: How many stories to generate per section (default 2).
    """
    sections = load_portfolio_sections()
    if section:
        if section not in sections:
            return f"Unknown section '{section}'. Options: {list(sections)}"
        sections = {section: sections[section]}

    written = []
    for sec, content in sections.items():
        stories = generate_stories_for_section(sec, content, count)
        for i, story_text in enumerate(stories, start=1):
            path = write_story_file(sec, i, story_text, "https://shaliniaiitd.github.io")
            written.append(str(path))
    return f"Wrote {len(written)} file(s): " + ", ".join(written) if written else "No stories generated."


@mcp.tool()
def reindex_vector_db(chunk_size: int = 60, overlap: int = 10) -> str:
    """(Re-)seed the vector store from every file under user_stories/,
    chunking each one and upserting with section metadata.

    Safe to call repeatedly -- it upserts, so existing entries get
    updated rather than duplicated. Call this after generate_user_stories
    adds new files.

    Args:
        chunk_size: Words per chunk.
        overlap: Word overlap between consecutive chunks.
    """
    seed.seed_from_user_stories(chunk_size=chunk_size, overlap=overlap)
    stats = collection_stats()
    return f"Reindexed. Vector store now has {stats['count']} chunk(s) across sections: {stats['sections']}"


@mcp.tool()
def search_portfolio(query: str, top_k: int = 3, section: str = "") -> str:
    """Search the portfolio vector store and return matches with their
    section, story id, and similarity distance (lower = more similar).

    Use this to answer questions like "what large-scale data work has she
    done" by retrieving the grounded chunks before answering, rather than
    guessing from general knowledge.

    Args:
        query: The search text.
        top_k: How many results to return.
        section: Optional filter, e.g. "projects", to restrict the search.
    """
    where = {"section": section} if section else None
    results = retrieve_similar_chunks(query, top_k=top_k, where=where)
    if not results:
        return "No results. Is the vector store seeded? Call reindex_vector_db first."
    lines = [
        f"[{r['metadata'].get('section', '?')} | distance={r['distance']:.4f}] {r['text']}"
        for r in results
    ]
    return "\n".join(lines)


@mcp.tool()
def run_rag_eval(top_k: int = 3) -> str:
    """Run the retrieval-quality eval (Hit@k across a fixed set of test
    queries grounded in the portfolio sections) and return a summary.

    Hit@k here means: for each test query, does a chunk from the expected
    section appear anywhere in the top_k retrieved results.
    """
    results = run_rag_evaluation(top_k=top_k)
    hits = sum(1 for r in results if r.hit)
    total = len(results)
    lines = [
        f"{'PASS' if r.hit else 'FAIL'}  expected={r.expected_section:<22} query={r.query}"
        for r in results
    ]
    lines.append(f"\nHit@{top_k}: {hits}/{total} = {hits / total:.0%}" if total else "No test cases.")
    return "\n".join(lines)


@mcp.tool()
def run_full_qa_workflow(user_story: str) -> str:
    """Run the entire QA workflow graph end-to-end: retrieve context,
    analyze, write dynamic memory, generate BDD, review, assemble report,
    and save it to disk. Returns the path to the saved report.

    Use this when you want the final report and don't need to inspect or
    steer intermediate steps. Use the individual tools instead if you want
    to review/adjust the analysis or BDD cases before generating the next
    artifact.

    Args:
        user_story: The raw user story text.
    """
    app = wf.build_graph()
    final_path = None
    for event in app.stream({"user_story": user_story}):
        if "save_report" in event:
            final_path = event["save_report"].get("output_path")
    return final_path or "No report path returned."


if __name__ == "__main__":
    mcp.run(transport="stdio")
