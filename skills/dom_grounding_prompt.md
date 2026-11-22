# DOM-grounded test generation skill

Convert the supplied BDD scenarios into one complete pytest-Playwright Python
script. Ground every assertion and selector in the scanned target page facts.

Target application URL: [[TARGET_APP_URL]]

REAL PAGE STRUCTURE (scanned from the live page for this run):
[[PAGE_FACTS_SUMMARY]]

BDD scenarios:
[[BDD_CASES]]

[[SYNTAX_FEEDBACK]]

## Grounding rules

- Do not invent selectors, CSS classes, navigation labels, content, metadata,
  links, routes, or application features that are not established by the page
  scan or the BDD scenario.
- The header brand is distinct from primary navigation. Use the scanned brand
  accessible name for the brand and only scanned navigation labels in the nav.
- Project content is rendered as articles under `#projects`; use the observed
  article structure and headings. Do not assume cards, tags, filters, modals,
  or a contact form.
- Only check metadata keys shown as present in the scan. Do not assume Open
  Graph tags exist.
- If a BDD scenario describes a feature not confirmed by the page scan, do not
  assert that a guessed control exists. Use a truthful negative/availability
  check when appropriate, or preserve the mismatch as an explicit test failure.
- An `alt` attribute with an empty value is valid for decorative images. Only
  a missing attribute (`get_attribute("alt") is None`) is an accessibility
  violation.
- Do not use exact load-time thresholds unless explicitly required by the BDD;
  arbitrary timing assertions are flaky.
- Do not load axe-core or other local/npm scripts unless their availability is
  established. Use installed dependencies only.

## Playwright correctness

- Use only the synchronous Playwright API, the built-in `page` fixture, plain
  `def test_...` functions, and `from playwright.sync_api import expect`.
- To inspect attributes across a locator, iterate `locator.all()` and call
  `.get_attribute()` on each element. `Locator.all_attribute_values()` does
  not exist.
- Use `exact=True` when matching a link by a complete accessible name unless
  substring matching is explicitly intended.
- For a positive count check, use `count = locator.count()` followed by
  `assert count > 0`; do not invent assertion helper names.
- Playwright actions and waits return `None` on success and raise on failure.
  Call them as statements; do not assert their return value.
- For external-link status checks, use this exact anti-bot-aware helper and
  call it for each status assertion:

```python
BOT_BLOCKING_DOMAINS = ("linkedin.com",)

def is_healthy_status(url: str, status: int) -> bool:
    if any(domain in url for domain in BOT_BLOCKING_DOMAINS):
        return status in (200, 999)
    return status == 200
```

## Output contract

- Include a short docstring on each test function mapping it to its BDD scenario.
- Return only the complete Python script, without markdown fences or explanatory
  prose.
- The script must be ready to run with `pytest <script_name> -v`.
