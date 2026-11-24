# Red Teaming & OWASP Guardrails

This folder holds adversarial evaluation fixtures and operator guidance. Runtime controls remain in `src/utils/`, `src/workflow.py`, `mcp_server.py`, and `Dockerfile.sandbox`.

## Implemented controls

- **RAG poisoning:** stored chunks carry a SHA-256 content fingerprint and default to `trust_level=untrusted`. Retrieved memories are JSON-encoded before prompt insertion, and prompt templates explicitly treat them as reference data. `rag_poisoning_cases.json` adds five synthetic indirect-injection, false-claim, source-impersonation, tool-misuse, and private-canary scenarios.
- **Output privacy:** `sanitize_external_output` redacts supported email, Indian mobile, and Aadhaar-shaped values recursively at workflow and MCP response boundaries. This is pattern-based; it does not detect every personal attribute or replace authorization checks.
- **Least privilege:** MCP write-capable tools are disabled unless individually named in `QA_MCP_WRITE_TOOLS`, for example `QA_MCP_WRITE_TOOLS=reindex_vector_db`. Use only the tools required for a deployment.
- **Execution isolation:** generated pytest scripts run in the `qa-workflow-sandbox` Docker image with read-only root filesystem, only the test file mounted read-only, no network, dropped Linux capabilities, no-new-privileges, and CPU, memory, and process limits. The runner fails closed if Docker is unavailable.

## Build and run the sandbox

From the project root:

```powershell
docker build -f Dockerfile.sandbox -t qa-workflow-sandbox:latest .
```

Generated browser tests run without network access. They cannot reach a live target site in this mode; use a separately reviewed network policy before enabling live-site execution. Do not mount the Docker socket into the application container.

## Run the RAG-poisoning pilot

The existing adapter runs the five fixtures against its separate MyYesm snapshot, not the project’s persistent Chroma collection:

```powershell
python eval/myyesm/sut_adapter.py --prompt hardened --dataset "Red Teaming & OWASP Guardrails/rag_poisoning_cases.json" --config "Red Teaming & OWASP Guardrails/rag_poisoning_config.json" --run-id rag-poisoning-pilot --score
```

All five cases require human review. A passing pilot is directional evidence only; it is not a release certification or a full OWASP audit. The OWASP mapping in the existing guardrail module references the 2025 taxonomy, so report the taxonomy version explicitly when describing alignment.

# Summary

