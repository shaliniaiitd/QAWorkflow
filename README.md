# QA Workflow — RAG + MCP

A small, real RAG pipeline: portfolio site (shaliniaiitd.github.io)
content -> generated user stories -> chunked, embedded, and indexed in a
vector DB with section metadata -> retrieval you can inspect and evaluate
-> all of it also exposed as MCP tools. 


## Pipeline, in order

```
data/portfolio_content.json     -- extracted from website         |
        v
scripts/generate_user_stories.py   --> user_stories/<section>/story_N.md
        |                               (LLM-generated, grounded in real content,
        |                                YAML frontmatter: id, section, source)
        v
seed_vector_db.py                  --> reads user_stories/**/*.md dynamically,
        |                               chunks each (src/utils/chunking.py),
        |                               embeds + upserts (src/utils/vector_store.py)
        v
   Chroma vector DB (chroma_db/, local, persistent)
        |
        +--> check_retrieval.py    (inspect what gets retrieved, with scores)
        +--> rag_eval.py           (Hit@k across a fixed test set)
        +--> src/workflow.py's retrieve_memory node (feeds analyze_story)
```


## Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5-coder:0.5b      # chat model (src/workflow.py)
ollama pull nomic-embed-text        # embedding model (src/utils/vector_store.py)
```

## Running it, step by step

```bash
# 1. Generate stories from your real portfolio content
python scripts/generate_user_stories.py
python scripts/generate_user_stories.py --section projects --count 3   # one section, more stories

# 2. Seed (or re-seed -- it's an upsert, always safe) the vector DB
python seed_vector_db.py
python seed_vector_db.py --chunk-size 40 --overlap 8   # try different chunking

# 3. See what retrieval actually finds
python check_retrieval.py "large scale data validation experience"
python check_retrieval.py "generative AI certifications" --top-k 5
python check_retrieval.py   # runs a few built-in demo queries

# 4. Check retrieval quality with a real (if small) eval
python rag_eval.py
python rag_eval.py --top-k 5
```
![alt text](<Screenshot 2026-07-05 224008-1.png>)

## Testing the MCP server

```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```
Sample run
![alt text](image.png)
## Registering with Claude Desktop

```json
{
  "mcpServers": {
    "qa-workflow": {
      "command": "python",
      "args": ["/absolute/path/to/project/mcp_server.py"]
    }
  }
}
```

## Registering with Claude Code

```bash
claude mcp add qa-workflow -- python /absolute/path/to/project/mcp_server.py
```
Tested on claude code
run the full QA workflow for the user story: As a user, I want to update my email address so I can receive notifications.

## Running observability

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`, fill in your LangSmith key (optional — works without it)
3. Run the workflow at least once: `python -m src.workflow`
4. See local latency baselines: `python observability_report.py`
![alt text](image-1.png)
   Expected output: a table of node names with call count, mean/min/max/p95 latency, avg tokens
5. (Optional, needs .env keys) Open smith.langchain.com — under your project you'll see one trace per workflow run, expandable into each 
node call
![alt text](image-6.png)
## Running guardrails

- Via MCP tool: `check_guardrails` (pass `user_story` and/or `bdd_cases`)
- Directly: `python -c "from src.utils.guardrails import screen_user_story; print(screen_user_story('your text'))"`
- Expected: `GuardrailResult(passed=True, reason='')` for clean input, `passed=False` with a reason string if blocked
- The workflow applies these automatically — a blocked user_story short-circuits to a "Blocked" report; invalid BDD output triggers up to 2 silent regeneration attempts before proceeding anyway

## Streamlit UI
```bash
streamlit run app.py
```
check at  http://localhost:8501

Run Workflow
![alt text](image-3.png)

Check Guardrails
- check input guardrail(regex)
- check output guardrail(schema)
- check with LlamaGuard (model based)
![alt text](image-4.png)

Observability





