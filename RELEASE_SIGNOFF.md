# Production trace coverage and executive release sign-off

## Production trace coverage

Set `APP_ENV=production`, `LANGSMITH_TRACING=true` (or `LANGCHAIN_TRACING_V2=true`), `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`), `LANGCHAIN_PROJECT`, and `RELEASE_ID` before starting the API. Production startup fails closed if LangSmith tracing is not enabled or the release id is missing. The API also requires `OBSERVABILITY_TOKEN` and protects `/api/observability` with bearer-token authentication. API request spans and child workflow spans receive `request_id`, `release_id`, environment, and service metadata. Request bodies, response bodies, query strings, and credentials are excluded from the API span and local request audit event. LLM trace inputs/outputs continue through the PII redactor.

`GET /api/observability` reports the fraction of retained API, Streamlit, and MCP workflow requests whose tracing was configured and includes local span IDs. The capture command checks those span IDs against LangSmith using the LangSmith API key; a GO packet requires the spans to be present there. Coverage is based on the most recent 200 events retained by the responding process. Multi-replica or separately hosted entry points need a shared/aggregated audit-event source before this number represents fleet-wide coverage. Protect this endpoint with the deployment's authentication/authorization layer and ship JSON logs to durable centralized storage. Local JSONL files are not a multi-replica retention system.

## Create the release evidence packet

Capture the live production coverage after representative traffic:

```powershell
python scripts/release_signoff.py capture-coverage --api-url https://qa-api.example.com --output release_signoffs\1.2.0\trace-coverage.json
```

Run the capture command in an operator environment with `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`) and `LANGCHAIN_PROJECT` configured for the same LangSmith project as production, and `OBSERVABILITY_TOKEN` set to the production endpoint token. It queries LangSmith for the captured request span IDs and records whether all were found before writing the evidence file.

Build a packet from current release evaluation artifacts (repeat `--evaluation` for each in-scope report). The MyYesm pilot is separate and should only be included if that assistant is part of this release:

```powershell
python scripts/release_signoff.py packet --release-id 1.2.0 --evaluation eval\qaworkflow\runs\ci-run-123.json --trace-coverage release_signoffs\1.2.0\trace-coverage.json
```

Review `decision_packet.md` and `.json`. Failed metrics, pending/failed human reviews, missing production traffic, incomplete trace configuration, and unverified LangSmith ingestion are blockers. Resolve each or explicitly accept it with a control action before recording GO.

## Record the decision

```powershell
python scripts/release_signoff.py sign --packet release_signoffs\1.2.0\decision_packet.json --approver-name "Name" --approver-title "VP Engineering" --decision no-go --rationale "Required release evidence is incomplete."
```

For GO, use `--decision go`; every listed blocker must be passed with `--accept-blocker <id>`, and every accepted risk requires a matching `--control-action "<id>=>action"` entry. The command records approver-supplied name/title, rationale, timestamp, packet digest, accepted blockers, and control actions. It does not authenticate the approver or create a cryptographic signature. Store the packet and sign-off in an access-controlled, immutable release evidence system and use your deployment platform's authenticated approval controls for identity verification.
