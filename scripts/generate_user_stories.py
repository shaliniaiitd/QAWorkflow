"""File: scripts/generate_user_stories.py

Generates realistic "user stories" grounded in Shalini's actual portfolio
content (data/portfolio_content.json), and writes each one to its own file
under user_stories/<section>/story_<n>.md.

Why generate from real content instead of hardcoding:
- The 3 hardcoded sample stories in the original workflow.py were generic
  placeholders, disconnected from any real source. Grounding generation in
  real portfolio text means the RAG index actually reflects something
  true, and retrieval quality means something (you can sanity-check "did
  it find the right section" against material you know).
- It also gives seed_vector_db.py something dynamic to read, per your
  request -- add a new section to portfolio_content.json, regenerate, and
  the vector DB picks it up on the next seed run without code changes.

Each generated story is framed as: "As a <visitor type>, I want to know
<something the section demonstrates> so that <why it matters to them>" --
i.e. a hiring-manager/recruiter perspective on the portfolio content, which
keeps it close enough to real QA-style user stories that generate_bdd/
review_bdd in workflow.py can still run against them meaningfully.

Output file format (Markdown with YAML frontmatter):

    ---
    id: work_experience_1
    section: work_experience
    source: https://shaliniaiitd.github.io
    ---
    As a hiring manager, I want to see evidence of large-scale data
    validation experience so that I can assess ETL testing depth.

How to run:
    python scripts/generate_user_stories.py
    python scripts/generate_user_stories.py --section projects --count 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.workflow import get_llm  # noqa: E402  (path insert must happen first)

PORTFOLIO_CONTENT_FILE = PROJECT_DIR / "data" / "portfolio_content.json"
USER_STORIES_DIR = PROJECT_DIR / "user_stories"

GENERATION_PROMPT = """You are helping build a small demo dataset of user stories.

Given the following section of a professional portfolio site, write {count} short
user stories from the point of view of someone evaluating the portfolio
(a hiring manager, recruiter, or technical interviewer). Each story should
follow the format:

As a <role>, I want to <goal grounded in the content below> so that <benefit>.

Section name: {section}
Section content:
\"\"\"
{content}
\"\"\"

Output exactly {count} stories, one per line, no numbering, no extra commentary.
"""


def _slugify_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().strip("-").strip()


def load_portfolio_sections() -> dict[str, str]:
    data = json.loads(PORTFOLIO_CONTENT_FILE.read_text(encoding="utf-8"))
    return data["sections"]


def generate_stories_for_section(section: str, content: str, count: int) -> list[str]:
    llm = get_llm()
    prompt = GENERATION_PROMPT.format(section=section, content=content, count=count)
    response = llm.invoke(prompt)
    raw_text = response.content if hasattr(response, "content") else str(response)
    lines = [
        _slugify_line(line)
        for line in raw_text.splitlines()
        if _slugify_line(line) and _slugify_line(line).lower().startswith("as a")
    ]
    return lines[:count]


def write_story_file(section: str, index: int, story_text: str, source_url: str) -> Path:
    section_dir = USER_STORIES_DIR / section
    section_dir.mkdir(parents=True, exist_ok=True)
    story_id = f"{section}_{index}"
    path = section_dir / f"story_{index}.md"
    frontmatter = (
        "---\n"
        f"id: {story_id}\n"
        f"section: {section}\n"
        f"source: {source_url}\n"
        "---\n"
    )
    path.write_text(frontmatter + story_text.strip() + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", default=None, help="Generate for one section only (default: all)")
    parser.add_argument("--count", type=int, default=2, help="Stories per section (default: 2)")
    args = parser.parse_args()

    data = json.loads(PORTFOLIO_CONTENT_FILE.read_text(encoding="utf-8"))
    source_url = data["source_url"]
    sections = data["sections"]

    if args.section:
        if args.section not in sections:
            raise SystemExit(f"Unknown section '{args.section}'. Options: {list(sections)}")
        sections = {args.section: sections[args.section]}

    written: list[Path] = []
    for section, content in sections.items():
        stories = generate_stories_for_section(section, content, args.count)
        if not stories:
            print(f"  [warn] no stories parsed for '{section}', skipping")
            continue
        for i, story_text in enumerate(stories, start=1):
            path = write_story_file(section, i, story_text, source_url)
            written.append(path)
            print(f"  wrote {path.relative_to(PROJECT_DIR)}")

    print(f"\nDone. Wrote {len(written)} story file(s) under {USER_STORIES_DIR.relative_to(PROJECT_DIR)}/")


if __name__ == "__main__":
    main()
