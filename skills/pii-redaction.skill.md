# Skill: PII redaction

## Purpose

Redact personal identifiers before text is logged, traced, or persisted.

## Scope

Apply this to:
- LangSmith trace payloads
- saved reports
- pytest execution logs
- any workflow output that may contain personal data

## Standard patterns

- emails: mask as `[REDACTED_EMAIL]`
- phone-like numbers: mask as `[REDACTED_PHONE]`
- Aadhaar-like identifiers: mask as `[REDACTED_ID]`

## Implementation reference

- [src/utils/guardrails.py](src/utils/guardrails.py)
- [src/utils/observability.py](src/utils/observability.py)

## Rule

Do not redact the workflow’s operational text unless it contains the known personal-data patterns. Keep console output readable; sanitize only persisted or exported artifacts.
