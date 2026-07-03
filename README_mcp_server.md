# QA Workflow — RAG + MCP

A small, real RAG pipeline: portfolio site (shaliniaiitd.github.io)
content -> generated user stories -> chunked, embedded, and indexed in a
vector DB with section metadata -> retrieval you can inspect and evaluate
-> all of it also exposed as MCP tools. 


## Pipeline, in order

```
data/portfolio_content.json
        |
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

## Testing the MCP server

```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```

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
