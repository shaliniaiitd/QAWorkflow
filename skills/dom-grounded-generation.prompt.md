---
mode: agent
description: Use the project’s DOM-grounding workflow for Playwright script generation.
---

# DOM-grounded generation

Use this when the task is to generate Playwright tests for the portfolio site from a real page scan instead of guessing selectors.

## Rules
- Inspect the real target page and collect nav links, headings, project blocks, external links, and image alt state.
- Prefer the DOM facts already captured in `scan_target_page()` and `_format_page_facts()` in `src/qa_workflow.py`.
- Do not invent selectors, CSS classes, or nav labels that are not in the live page facts.
- Prefer verified Playwright sync APIs; avoid nonexistent helpers or unsupported methods.
- Keep assertions grounded in the page structure and avoid arbitrary timing thresholds.

## Project references
- [src/qa_workflow.py](src/qa_workflow.py)
- [skills/dom_grounding_prompt.md](skills/dom_grounding_prompt.md)
- [README.md](README.md)

## Output contract
- Generate only a complete pytest-playwright Python script.
- Use synchronous API patterns.
- Include docstrings mapping each test back to the BDD scenario.
- Return no markdown fences and no explanatory prose.
