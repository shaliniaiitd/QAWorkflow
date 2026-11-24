# Python Playwright page objects

Run the deterministic local page-object regression with `python -m pytest tests/ui -q` after installing requirements and `playwright install chromium`.

`tests/ui/pages/` holds shared page objects. Tests use role and accessible-name locators, and the fixture is local so CI has no dependency on the public portfolio site. For a real deployment smoke test, add an environment-specific test that reads an explicitly configured base URL; do not make pull request CI depend on a public site.

## Playwright MCP

The repository `.mcp.json` registers Microsoft's official `@playwright/mcp` server for MCP-compatible clients such as Claude Code. Install Node.js 20+ and Chromium first. The server is isolated and headless, and restricts browser requests to the portfolio and local API origins. The origin list is a convenience control, not a security boundary. MCP browser inspection is an agent execution surface; CI runs deterministic pytest-playwright tests separately.
