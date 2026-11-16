# Skill: QA test generation checklist

## Purpose

Generate reliable Playwright tests from BDD cases and grounded project facts.

## Checklist

- Use synchronous pytest-playwright APIs only.
- Require real `page` fixture usage.
- Ensure generated tests follow `def test_...()` naming.
- Validate Python syntax before saving the script.
- Prefer known page facts over guessed selectors.
- Avoid unsupported Playwright methods and fake assertion helpers.
- Use real project structure (nav, headings, articles, external links) rather than invented page assumptions.
- Separate phase 1 and phase 2 artifacts by directory.

## Reference

- [src/qa_workflow.py](src/qa_workflow.py)
- [QAWorkflow_Design_doc.md](QAWorkflow_Design_doc.md)
