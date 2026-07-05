"""File: src/utils/guardrails.py

LLM guardrails: validating what goes INTO the model and what comes OUT of
it. Four checks now, each targeting a different risk:

1. check_prompt_injection() -- INPUT. Prompt injection is the LLM-native
   analog of SQL injection: a string trying to override instructions
   instead of a query. Pattern-based (see module note on limits).

2. check_sql_injection_pattern() -- INPUT. Classic SQLi shapes, kept for
   completeness -- different discipline from LLM guardrails (see chat).

3. check_bdd_output_schema() -- OUTPUT. Confirms generated BDD has real
   Gherkin structure before it's used downstream. Wired into workflow.py's
   validate_bdd -> retry loop.

4. redact_pii() -- OUTPUT/logging. Not a safety check that blocks
   anything -- a data-minimization helper. Strips emails, phone numbers,
   and Aadhaar-shaped 12-digit numbers from text before it's logged or
   stored, relevant under India's DPDP Act 2023 (data minimization: don't
   retain personal data you don't need to). Note this project's current
   observability logging already only stores character counts, not raw
   text (see observability.py) -- redact_pii is here for if/when 
   choice is made to log actual prompt/response content for debugging.

5. check_with_llamaguard() -- OUTPUT/INPUT, OPTIONAL. Runs Meta's
   Llama Guard model (via Ollama) as a second, model-based safety
   classifier, instead of the regex heuristics above. Requires
   `ollama pull llama-guard3` first; gracefully no-ops (passed=True, with
   a note in the reason) if the model isn't available, so calling it never
   breaks anything if you haven't set it up.

--- Mapping to OWASP Top 10 for LLM Applications (2025) ---
Not exhaustive -- this maps what's covered here, and flags what isn't so
you don't accidentally claim more coverage than exists:

  LLM01 Prompt Injection            -> check_prompt_injection, check_with_llamaguard
  LLM02 Sensitive Information Disclosure -> redact_pii (mitigates, doesn't detect on the way out of the LLM)
  LLM05 Improper Output Handling    -> check_bdd_output_schema
  LLM06 Excessive Agency            -> partially: MCP tool annotations (Destructive/Idempotent
                                        flags in mcp_server.py) scope what each tool is allowed to do
  LLM04 Data/Model Poisoning        -> NOT covered here (would apply to training your own model, not this project)
  LLM03 Supply Chain Vulnerabilities -> NOT covered here (would mean auditing model/package provenance)
  LLM07 System Prompt Leakage       -> partially: check_prompt_injection catches the common
                                        "reveal your system prompt" phrasing, not sophisticated extraction
  LLM08 Vector/Embedding Weaknesses -> NOT covered here (would mean auditing vector_store.py
                                        for embedding-inversion or poisoned-document attacks)
  LLM09 Misinformation              -> loosely related to rag_eval.py's Hit@k, but that measures
                                        retrieval accuracy, not hallucination in generated text
  LLM10 Unbounded Consumption       -> NOT covered here (would mean rate-limiting/cost caps on LLM calls)

Being able to say "here's what I cover and here's what I know I don't"

"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Heuristic phrases that commonly appear in prompt injection attempts.
# Real systems combine this kind of pattern list with an LLM-based judge
# for anything more sophisticated than these obvious cases.
_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"you are now",
    r"new instructions?:",
    r"system\s*:",
    r"reveal (your|the) (system )?prompt",
    r"act as (if you|though) you (have no|are not) restrictions",
    r"jailbreak",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# SQL-injection-style patterns. Not because this project talks to a SQL
# database with user_story text (it doesn't), but because if you ever DO
# pass user-supplied text into a query builder, filename, or shell command
# downstream of an agent, checking for this shape of input is standard
# defensive practice -- worth having the pattern ready.
_SQLI_PATTERNS = [
    r"'\s*or\s*'1'\s*=\s*'1",
    r";\s*drop\s+table",
    r"union\s+select",
    r"--\s*$",
    r"xp_cmdshell",
]
_SQLI_RE = re.compile("|".join(_SQLI_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""


def check_prompt_injection(text: str) -> GuardrailResult:
    """Flag text that looks like it's trying to override system instructions.

    Args:
        text: Untrusted input, e.g. a user_story before it reaches an LLM.
    """
    match = _INJECTION_RE.search(text)
    if match:
        return GuardrailResult(passed=False, reason=f"possible prompt injection: matched {match.group(0)!r}")
    return GuardrailResult(passed=True)


def check_sql_injection_pattern(text: str) -> GuardrailResult:
    """Flag text with classic SQL-injection shapes.

    Defensive-input-validation pattern, included for completeness -- see
    the module docstring for why this is a different discipline from LLM
    guardrails proper.
    """
    match = _SQLI_RE.search(text)
    if match:
        return GuardrailResult(passed=False, reason=f"possible SQL injection pattern: matched {match.group(0)!r}")
    return GuardrailResult(passed=True)


def check_bdd_output_schema(bdd_text: str) -> GuardrailResult:
    """Validate that generated BDD text has real Gherkin structure.

    Minimum bar: non-trivial length, and contains at least one Given/When/
    Then keyword (case-insensitive). This won't catch subtly wrong BDD --
    it catches the LLM returning something empty, truncated, or completely
    off-format, which is the failure mode worth gating on automatically.
    """
    if not bdd_text or len(bdd_text.strip()) < 20:
        return GuardrailResult(passed=False, reason="output too short or empty")

    has_gherkin_keyword = bool(re.search(r"\b(given|when|then)\b", bdd_text, re.IGNORECASE))
    if not has_gherkin_keyword:
        return GuardrailResult(passed=False, reason="no Given/When/Then structure found")

    return GuardrailResult(passed=True)


def screen_user_story(user_story: str) -> GuardrailResult:
    """Run all input-side checks on a user story before it reaches the LLM."""
    for check in (check_prompt_injection, check_sql_injection_pattern):
        result = check(user_story)
        if not result.passed:
            return result
    return GuardrailResult(passed=True)


# --- DPDP Act 2023-oriented: data minimization for logging ---

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b[6-9]\d{9}\b")  # Indian 10-digit mobile pattern
_AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")  # 12-digit, optionally spaced


def redact_pii(text: str) -> str:
    """Strip common personal-data patterns before logging/storing text.

    Data minimization -- don't retain personal data you don't need to --
    is a core principle under India's DPDP Act 2023. This is a heuristic
    redactor (emails, Indian mobile numbers, Aadhaar-shaped 12-digit
    numbers), not a certified compliance tool. Use it before writing raw
    prompt/response text to any log or store; this project's own
    observability log doesn't currently store raw text at all (see
    observability.py), so this is here for when you do.
    """
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _AADHAAR_RE.sub("[REDACTED_ID]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text


# --- Optional: Llama Guard as a second, model-based classifier ---

def check_with_llamaguard(text: str, base_url: str = "http://localhost:11434") -> GuardrailResult:
    """Classify text as safe/unsafe using Meta's Llama Guard model via Ollama.

    Requires `ollama pull llama-guard3` first. This is a MODEL-based check
    (the model reasons about intent/content) as opposed to the regex-based
    checks above (which only match known phrasings) -- meaningfully
    different coverage, at the cost of an extra LLM call's latency.

    If the model isn't available (not pulled, Ollama unreachable), this
    returns passed=True with a note explaining it was skipped -- calling
    it is always safe, it just won't add protection until you've pulled
    the model.
    """
    try:
        from langchain_ollama import ChatOllama

        guard = ChatOllama(model="llama-guard3", temperature=0, base_url=base_url)
        response = guard.invoke(text)
        content = (response.content if hasattr(response, "content") else str(response)).strip().lower()
        if content.startswith("unsafe"):
            return GuardrailResult(passed=False, reason=f"LlamaGuard flagged as unsafe: {content}")
        return GuardrailResult(passed=True)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure here should not block the pipeline
        return GuardrailResult(passed=True, reason=f"LlamaGuard unavailable, check skipped ({exc})")