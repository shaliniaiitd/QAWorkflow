"""File: seed_vector_db.py (replaces the seeding half of run_demo.py)

Dynamically seeds the vector DB from every file under user_stories/,
instead of a hardcoded list. This is the "part of the flow vs. separate
script" question resolved: seeding stays a separate, explicit step (run
on demand, not on every workflow execution) -- but it's no longer tied to
three hardcoded strings. Run scripts/generate_user_stories.py to add more
source material, then re-run this to (re-)index it.

What changed vs. the original run_demo.py's seed_vector_db():
- Reads user_stories/<section>/*.md dynamically (glob), instead of three
  add_story_to_db(...) calls.
- Parses the YAML frontmatter each file was written with (id, section,
  source) so that metadata -- not just raw text -- goes into the vector
  store. That's what makes section-filtered retrieval and the RAG eval
  script possible.
- Chunks each story's body with src.utils.chunking.chunk_text before
  embedding. For these short one-paragraph stories a single chunk is
  common, but the pipeline is ready for longer source documents (e.g. if
  you later feed it full project write-ups instead of one-line stories)
  without any code changes here.
- Upserts (via vector_store.add_story_to_db, which now does
  collection.upsert) so re-running this is always safe.

How to run:
    python seed_vector_db.py
    python seed_vector_db.py --chunk-size 40 --overlap 8
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.utils.chunking import chunk_text
from src.utils.vector_store import add_story_to_db, collection_stats

PROJECT_DIR = Path(__file__).resolve().parent
USER_STORIES_DIR = PROJECT_DIR / "user_stories"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_story_file(path: Path) -> tuple[dict[str, str], str]:
    """Parse the simple `key: value` YAML frontmatter written by
    generate_user_stories.py, plus the body text below it.

    Not a general YAML parser -- deliberately minimal, matching exactly
    the frontmatter shape this project writes. If you introduce nested
    YAML later, swap this for `python-frontmatter` or `pyyaml`.
    """
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw.strip()
    frontmatter_block, body = match.groups()
    metadata = {}
    for line in frontmatter_block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return metadata, body.strip()


def seed_from_user_stories(chunk_size: int = 60, overlap: int = 10) -> None:
    if not USER_STORIES_DIR.exists():
        raise SystemExit(
            f"{USER_STORIES_DIR} does not exist yet. Run "
            "scripts/generate_user_stories.py first."
        )

    story_files = sorted(USER_STORIES_DIR.glob("**/*.md"))
    if not story_files:
        raise SystemExit(f"No .md files found under {USER_STORIES_DIR}.")

    total_chunks = 0
    for path in story_files:
        metadata, body = parse_story_file(path)
        story_id = metadata.get("id", path.stem)
        section = metadata.get("section", path.parent.name)
        source = metadata.get("source", "")

        chunks = chunk_text(body, chunk_size=chunk_size, overlap=overlap)
        for chunk in chunks:
            chunk_id = f"{story_id}::chunk{chunk.chunk_index}"
            add_story_to_db(
                chunk_id,
                chunk.text,
                metadata={
                    "story_id": story_id,
                    "section": section,
                    "source": source,
                    "chunk_index": chunk.chunk_index,
                },
            )
            total_chunks += 1
        print(f"  seeded {path.relative_to(PROJECT_DIR)} -> {len(chunks)} chunk(s)")

    print(f"\nSeeded {len(story_files)} file(s) as {total_chunks} chunk(s).")
    print("Collection stats:", collection_stats())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=60, help="Words per chunk (default: 60)")
    parser.add_argument("--overlap", type=int, default=10, help="Word overlap between chunks (default: 10)")
    args = parser.parse_args()
    seed_from_user_stories(chunk_size=args.chunk_size, overlap=args.overlap)


if __name__ == "__main__":
    main()
