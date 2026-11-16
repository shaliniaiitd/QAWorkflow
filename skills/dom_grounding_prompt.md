# Skill: DOM-grounded test generation

Use this pattern whenever the workflow must generate Playwright tests for a live application with a known URL.

## Purpose

Ground the LLM in the actual DOM instead of letting it invent selectors, nav labels, classes, or page structure. This reduces false positives, missing selectors, and unsupported Playwright API guesses.

## Required inputs

- target URL
- one or more BDD scenarios
- a real page scan result containing:
  - navigation link text in DOM order
  - headings
  - project/article blocks
  - page title and metadata
  - external link list
  - class sample
  - image alt-text state

## Decision rules

- Do not invent selectors or classes that do not appear in the scan.
- Prefer real, verified DOM nodes from the scan.
- Distinguish the header brand from nav links.
- Treat an empty `alt` attribute as valid for decorative images.
- Do not assert exact timing thresholds unless the scenario explicitly requires them.
- Never use unsupported API names or patterns that were not verified on the target Playwright version.
- Use plain `assert` with `locator.count()` rather than guessing at non-existent assertion helpers.

## Output contract

Return only a complete pytest-playwright Python script:

- synchronous API only
- `def test_...()` functions only
- `from playwright.sync_api import expect`
- single script file with no markdown fences and no explanation text

## Prompt template

```text
Convert the following BDD Gherkin test cases into a pytest-playwright Python script.

The target application is at: {TARGET_APP_URL}

REAL PAGE STRUCTURE (scanned directly from the live page):
{PAGE_FACTS_SUMMARY}

Rules:
- use the real structure above; do not invent selectors or nav labels
- only assert things confirmed by the real page scan
- use the exact external-link status helper pattern provided below
- use Playwright sync methods only
- avoid unsupported APIs or guessed fluent assertion names
- no timing assertions unless the scenario requires them

BDD Cases:
{BDD_CASES}

Generate ONLY the complete Python script.
```

## Typical usage

This is the reusable skill behind the Phase 2a flow in `src/qa_workflow.py`, where `scan_target_page()` feeds a DOM-grounded prompt into the generated script workflow.
