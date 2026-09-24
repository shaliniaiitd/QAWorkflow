"""File: src/qa_workflow.py

Phase 1 of QAWorkflow — Multi-Agent Architecture with HITL Approval Gates

This refactors the linear pipeline (workflow.py) into a hybrid supervisor + swarm
topology with human-in-the-loop gates at three decision points:

  1. Before ExecutionAgent runs tests (HITL Gate 2)
  2. Before HealingAgent applies a fix (HITL Gate 1)  
  3. Before final report is published (HITL Gate 3)

Architecture (see design doc Section 6 for full details):
  - Supervisor: entry point, owns screen_input guardrail + run metadata (run_id, timestamps)
  - TestCaseGenAgent: swarm peer, wraps analyze_story → write_dynamic_memory → 
                      generate_bdd → validate_bdd (schema retry loop)
  - TestCaseReviewAgent: swarm peer, qualitative review + assigns test_type tag
                         (smoke/regression/sanity/exploratory), can swarm back to Gen on fail
  - ExecutionAgent: swarm peer, Phase 1 generates Playwright script and runs it
  - HealingAgent: swarm peer, diagnoses failures, proposes fix, interrupts for HITL approval
  - ReportAgent: swarm peer, aggregates results, interrupts for final HITL approval

Swarm handoff decisions are CODE-BASED, not LLM-tool-call-based (see design doc
Section 6.1 and Section 12.1 on model capacity risks).

HITL mechanism: LangGraph's interrupt() + Command(resume=...) backed by checkpointer.
Phase 1 uses CLI input() for approval; Phase 2 moves to app.py / LangGraph Studio / MCP.

Prereqs (same as workflow.py):
  - Ollama running on http://localhost:11434
  - Models available: qwen2.5-coder:0.5b (chat), nomic-embed-text (embeddings)
  - ChromaDB vector store seeded at data/vectorstore/
  - Prompts at prompts/*.txt
  - project_memory.json at project root

How to run:
    python -m src.qa_workflow
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import TypedDict, cast, Literal
from datetime import datetime

AUTO_APPROVE = os.environ.get("AUTO_APPROVE", "0").lower() in {"1", "true", "yes", "y"}

from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig
from src.utils.vector_store import retrieve_similar_stories
from src.utils.observability import traceable, timed_call, log_llm_call
from src.utils.guardrails import screen_user_story, check_bdd_output_schema
from seed_vector_db import parse_story_file  # reuse the same frontmatter parser seed_vector_db.py uses

try:
    from langgraph.graph import END, StateGraph, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: langgraph. Install with: pip install langgraph"
    ) from exc


MODEL_CONFIG = {
    "model_name": os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-20b"),
    "temperature": 0.2,
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
}

TARGET_APP_URL = os.environ.get("TARGET_APP_URL", "https://shaliniaiitd.github.io")

PROJECT_DIR = Path(__file__).resolve().parents[1]
MEMORY_FILE = PROJECT_DIR / "project_memory.json"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUT_FILE = OUTPUTS_DIR / "sample_report.md"

PROMPT_FIELDS = {
    "analysis": ["static_memory", "retrieved_context", "user_story"],
    "dynamic_memory": ["static_memory", "analysis"],
    "bdd": ["static_memory", "dynamic_memory", "user_story", "analysis"],
    "review": ["static_memory", "dynamic_memory", "bdd_cases"],
    "report": ["user_story", "static_memory", "retrieved_context", "dynamic_memory", "analysis", "bdd_cases", "review_notes"],
}

MAX_BDD_RETRIES = 2
MAX_HEALING_ATTEMPTS = 3  # Max number of heal→approval→execute loops before giving up


class QAState(TypedDict, total=False):
    """Extended state schema for Phase 1 supervisor + swarm topology.
    
    See design doc Section 7 for the full schema contract.
    """
    # Inputs
    user_story: str

    # Memory
    static_memory: str
    dynamic_memory: str
    retrieved_context: str

    # Artifacts from gen/review
    analysis: str
    bdd_cases: str
    review_notes: str
    test_type: str  # smoke | regression | sanity | exploratory

    # Execution results
    execution_result: str  # pass | fail
    execution_log: str
    healing_proposed: str  # description of proposed fix
    healing_applied: bool
    healing_attempts: int  # number of heal→approve→execute cycles attempted

    # Final output
    final_report: str
    output_path: str

    # Guardrails / intelligent-workflow control
    input_blocked: bool
    block_reason: str
    bdd_valid: bool
    retry_count: int

    # Phase 1: HITL control
    hitl_gate: str  # "pre_execution" | "pre_healing" | "pre_publish" | "" (none pending)
    hitl_decision: str  # "approve" | "reject" | "edit"

    # Metadata
    run_id: str  # stamped by Supervisor, unique per run
    hops: int  # swarm loop safety counter


def create_llm():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        return ChatGroq(
            api_key=groq_api_key,
            model=MODEL_CONFIG["model_name"],
            temperature=MODEL_CONFIG["temperature"],
            max_tokens=4000,  # headroom for codegen output on top of reasoning tokens
            reasoning_effort="low",  # openai/gpt-oss-20b spends part of its token budget on
                                     # hidden reasoning before the final answer; "medium" (the
                                     # default) was eating enough of that budget on longer/more
                                     # complex prompts (e.g. Phase 2a's grounded codegen) to
                                     # squeeze the final content out entirely, returning empty.
                                     # "low" is appropriate here since codegen-from-structured-
                                     # input doesn't need heavy deliberation.
        )

    return ChatOllama(
        model=MODEL_CONFIG["model_name"],
        temperature=MODEL_CONFIG["temperature"],
        base_url=MODEL_CONFIG["base_url"],
    )


@traceable(name="llm_text_call")
def _llm_text(llm, prompt: str, node_name: str = "unknown") -> str:
    with timed_call() as t:
        msg = llm.invoke(prompt)
    raw_content = msg.content if hasattr(msg, "content") else msg
    text = raw_content if isinstance(raw_content, str) else str(raw_content)
    usage = getattr(msg, "usage_metadata", None)
    log_llm_call(node_name, t.elapsed, len(prompt), len(text), usage)
    return text


_LLM = None


def get_llm():
    """Lazily construct (and cache) the LLM client."""
    global _LLM
    if _LLM is None:
        _LLM = create_llm()
    return _LLM


# Utility functions (same as workflow.py)

def _format_static_memory(memory: dict) -> str:
    common_risks = ", ".join(memory.get("common_risks", []))
    known_bug_patterns = ", ".join(memory.get("known_bug_patterns", []))
    return (
        f"Project name: {memory.get('project_name', 'Unknown')}\n"
        f"BDD style: {memory.get('bdd_style', 'Default')}\n"
        f"Common risks: {common_risks}\n"
        f"Known bug patterns: {known_bug_patterns}\n"
    )


def load_static_memory(state: QAState) -> QAState:
    if not MEMORY_FILE.exists():
        raise FileNotFoundError(f"Missing {MEMORY_FILE}.")
    memory = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {"static_memory": _format_static_memory(memory)}


def load_prompt(name: str) -> str:
    path = PROJECT_DIR / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def format_prompt(template_name: str, state: QAState) -> str:
    template = load_prompt(template_name)
    fields = {field: state.get(field, "") for field in PROMPT_FIELDS[template_name]}
    return template.format(**fields)


# ============================================================================
# SUPERVISOR NODE — Entry point, owns guardrail + run metadata
# ============================================================================

def supervisor(state: QAState) -> QAState:
    """Entry point: screen input, stamp run metadata, route based on guardrail."""
    result = screen_user_story(state.get("user_story", ""))
    
    # Stamp run_id and initial hops counter once, at Supervisor level
    run_id = state.get("run_id") or str(uuid.uuid4())[:8]

    return {
        "input_blocked": not result.passed,
        "block_reason": result.reason,
        "run_id": run_id,
        "hops": state.get("hops", 0),
    }


def route_after_supervisor(state: QAState) -> str:
    """Supervisor routes based on input guardrail result."""
    if state.get("input_blocked"):
        return "blocked_report"
    return "retrieve_memory"


# ============================================================================
# BLOCKED REPORT — Short-circuit for guardrail failures
# ============================================================================

def blocked_report(state: QAState) -> QAState:
    reason = state.get("block_reason", "unspecified guardrail violation")
    report = (
        "# QA Report -- Blocked\n\n"
        "The submitted user story was blocked by an input guardrail.\n\n"
        f"Reason: {reason}\n"
    )
    return {"final_report": report}


# ============================================================================
# TESTCASEGENAGENT — Wraps analysis → generation → schema validation retry
# ============================================================================

def retrieve_memory(state: QAState) -> QAState:
    """Retrieve similar past stories via RAG."""
    query = state.get("user_story", "")
    similar = retrieve_similar_stories(query, top_k=3)
    return {"retrieved_context": "\n".join(similar)}


@traceable(name="analyze_story")
def analyze_story(state: QAState) -> QAState:
    prompt = format_prompt("analysis", state)
    return {"analysis": _llm_text(get_llm(), prompt, node_name="analyze_story")}


@traceable(name="write_dynamic_memory")
def write_dynamic_memory(state: QAState) -> QAState:
    prompt = format_prompt("dynamic_memory", state)
    return {"dynamic_memory": _llm_text(get_llm(), prompt, node_name="write_dynamic_memory")}


@traceable(name="generate_bdd")
def generate_bdd(state: QAState) -> QAState:
    prompt = format_prompt("bdd", state)
    return {"bdd_cases": _llm_text(get_llm(), prompt, node_name="generate_bdd")}


def validate_bdd(state: QAState) -> QAState:
    """Output guardrail: schema validation with retry loop."""
    result = check_bdd_output_schema(state.get("bdd_cases", ""))
    retry_count = state.get("retry_count", 0)
    if not result.passed:
        retry_count += 1
    return {"bdd_valid": result.passed, "retry_count": retry_count}


def route_after_validate_bdd(state: QAState) -> str:
    """Decide: retry generate, or proceed to review?"""
    if state.get("bdd_valid"):
        return "test_case_review_agent"
    if state.get("retry_count", 0) >= MAX_BDD_RETRIES:
        # Stop retrying, proceed anyway
        return "test_case_review_agent"
    return "generate_bdd"


# ============================================================================
# TESTCASEREVIEWAGENT — Qualitative review + test_type tagging
# ============================================================================

@traceable(name="review_bdd")
def review_bdd(state: QAState) -> QAState:
    """Qualitative review of generated BDD."""
    prompt = format_prompt("review", state)
    return {"review_notes": _llm_text(get_llm(), prompt, node_name="review_bdd")}


def assign_test_type(state: QAState) -> QAState:
    """Rule-based assignment of test_type (smoke/regression/sanity/exploratory).
    
    Phase 1: simple rule-based logic. Phase 2+: can add LLM tiebreaker for ambiguous cases.
    """
    # TODO: implement real logic based on story history, flow criticality, etc.
    # For now, default to "sanity" as a placeholder
    test_type = "sanity"
    return {"test_type": test_type}


def test_case_review_agent(state: QAState) -> QAState:
    """Wraps review_bdd + test_type assignment."""
    # Run review
    state = review_bdd(state)
    # Assign type
    state = assign_test_type(state)
    return state


def route_after_review(state: QAState) -> str:
    """Review agent can swarm back to gen, or forward to execution."""
    # TODO: code-based decision on whether review is satisfied
    # For now, always proceed to execution
    return "pre_execution_hitl"


# ============================================================================
# EXECUTION HITL GATE — Before ExecutionAgent runs
# ============================================================================

def pre_execution_hitl(state: QAState) -> QAState:
    """HITL Gate 2: Approve before test execution?"""
    print("\n" + "="*70)
    print("HITL GATE 2: Pre-Execution Approval")
    print("="*70)
    print(f"\nUser Story: {state.get('user_story', '')[:100]}...")
    print(f"Test Type: {state.get('test_type', 'unknown')}")
    print(f"BDD Cases:\n{state.get('bdd_cases', '')[:200]}...")
    if AUTO_APPROVE:
        print("\nAuto-approve enabled: approving execution.")
        decision = "approve"
    else:
        print("\nDo you approve execution? (Y/N): ", end="", flush=True)
        user_input = input().strip().lower()
        decision = "approve" if user_input in {"y", "yes"} else "reject"

    return {
        "hitl_gate": "pre_execution",
        "hitl_decision": decision,
    }


def route_after_pre_execution_hitl(state: QAState) -> str:
    if state.get("hitl_decision") == "approve":
        return "execution_agent"
    else:
        # Reject: skip to report with no execution
        return "report_agent"


# ============================================================================
# EXECUTIONAGENT — Phase 1: Scripted execution (stub)
# ============================================================================

def _generate_playwright_script(bdd_cases: str, node_name: str) -> str:
    """Shared codegen call: BDD Gherkin -> pytest-playwright Python script.
    
    Used by both execution_agent (interactive --demo run) and
    generate_script_for_story (batch mode), so the prompt only lives once.
    """
    prompt = f"""Convert the following BDD Gherkin test cases into a pytest-playwright Python script.

The target application under test is at: {TARGET_APP_URL}
Use this EXACT URL as the base URL in the script (e.g. BASE_URL = "{TARGET_APP_URL}") --
do NOT invent a placeholder URL like example.com.

The script MUST:
- Use the SYNCHRONOUS pytest-playwright API (NOT async/await, NOT playwright.async_api)
- Use the built-in `page` fixture provided automatically by the pytest-playwright plugin
  (no manual browser/context setup, no custom fixtures needed)
- Follow pytest naming conventions (test_* functions), using plain `def`, never `async def`
- Call Playwright's synchronous methods directly on `page` (e.g. page.goto(), page.fill(),
  page.click(), page.locator(...).click())
- Use `from playwright.sync_api import expect` for assertions where appropriate
- Include a short docstring on each test function mapping it back to its BDD scenario
- Be ready to run as-is with: pytest <script_name> -v

BDD Cases:
{bdd_cases}

Generate ONLY the complete Python script -- no explanations, no markdown code fences."""
    raw = _llm_text(get_llm(), prompt, node_name=node_name)
    return _strip_code_fences(raw)


def _strip_code_fences(text: str) -> str:
    """Defensively strip markdown code fences the model may add despite instructions
    not to -- writing a fenced block directly to a .py file breaks the script."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _run_pytest_script(script_path: Path, log_name: str, timeout: int = 120, phase: str = "phase1") -> tuple[str, str, Path]:
    """Run a pytest script, streaming output live to console while also
    saving the full output to outputs/test_results/<phase>/<log_name>.log.

    Shared by execution_agent (--demo mode) and run_batch (both phase1 and
    phase2), so console visibility + persisted logs happen the same way
    everywhere, just filed under the right phase folder for comparison.

    Returns (execution_result, execution_log_summary, log_file_path).
    """
    results_dir = OUTPUTS_DIR / "test_results" / phase
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{log_name}.log"

    lines: list[str] = []
    returncode = -1
    try:
        process = subprocess.Popen(
            ["pytest", str(script_path), "-v", "--tb=short"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        start = time.time()
        for line in process.stdout:
            print(line, end="")  # live on console
            lines.append(line)
            if time.time() - start > timeout:
                process.kill()
                lines.append(f"\n[Timeout] Killed after {timeout}s.\n")
                break
        process.wait(timeout=5)
        returncode = process.returncode
    except FileNotFoundError:
        lines.append("pytest not found. Ensure pytest and pytest-playwright are installed.\n")
    except Exception as e:
        lines.append(f"Error running pytest: {e}\n")

    execution_result = "pass" if returncode == 0 else "fail"
    full_log = "".join(lines)
    log_path.write_text(full_log, encoding="utf-8")

    summary = (
        f"Pytest completed with return code {returncode} ({execution_result}). "
        f"Full log: {log_path.relative_to(PROJECT_DIR)}"
    )
    return execution_result, summary, log_path


def execution_agent(state: QAState) -> QAState:
    """Phase 1: Generate pytest-playwright script from BDD and run it.
    
    Flow:
    1. Use LLM to convert BDD cases to pytest-playwright Python code
    2. Save script to tests/test_generated_<run_id>.py
    3. Run via pytest (streamed live to console, full log saved to disk)
    4. Determine pass/fail from pytest's return code
    
    In Phase 2, this becomes live/scriptless tool-calling (Playwright MCP).
    """
    print("\n[ExecutionAgent] Generating test script from BDD...")
    
    run_id = state.get("run_id", "unknown")
    bdd_cases = state.get("bdd_cases", "")
    
    script_code = _generate_playwright_script(bdd_cases, node_name="execution_agent_codegen")
    
    # Save script to tests/phase1/ folder (Phase 2 will write to tests/phase2/,
    # same story id, so before/after scripts sit side by side -- see design doc 9.1.1)
    tests_dir = PROJECT_DIR / "tests" / "phase1"
    tests_dir.mkdir(parents=True, exist_ok=True)
    script_path = tests_dir / f"test_generated_{run_id}.py"
    
    script_path.write_text(script_code, encoding="utf-8")
    print(f"✓ Test script saved to {script_path}")
    
    print(f"✓ Running tests via pytest...")
    execution_result, execution_log, log_path = _run_pytest_script(script_path, log_name=f"run_{run_id}")
    
    print(f"✓ Execution result: {execution_result} (full log: {log_path})")
    return {
        "execution_result": execution_result,
        "execution_log": execution_log,
    }


def route_after_execution(state: QAState) -> str:
    if state.get("execution_result") == "pass":
        return "report_agent"
    else:
        return "pre_healing_hitl"


# ============================================================================
# HEALING HITL GATE — Before HealingAgent applies a fix
# ============================================================================

def pre_healing_hitl(state: QAState) -> QAState:
    """HITL Gate 1: Approve healing fix before applying?
    
    Also checks: have we exceeded MAX_HEALING_ATTEMPTS? If so, skip healing and go to report.
    """
    healing_attempts = state.get("healing_attempts", 0)
    
    if healing_attempts >= MAX_HEALING_ATTEMPTS:
        print("\n" + "="*70)
        print(f"HEALING ATTEMPT LIMIT REACHED ({MAX_HEALING_ATTEMPTS} attempts)")
        print("="*70)
        print(f"\nGiving up on automatic healing after {MAX_HEALING_ATTEMPTS} attempts.")
        print("Proceeding to final report without further healing.")
        return {
            "hitl_gate": "pre_healing_limit_reached",
            "hitl_decision": "reject",
        }
    
    print("\n" + "="*70)
    print("HITL GATE 1: Pre-Healing Approval")
    print(f"(Healing attempt {healing_attempts + 1}/{MAX_HEALING_ATTEMPTS})")
    print("="*70)
    print(f"\nExecution Failed. Diagnosis:")
    print(f"{state.get('healing_proposed', 'No diagnosis yet')}")
    if AUTO_APPROVE:
        print("\nAuto-approve enabled: approving healing fix.")
        decision = "approve"
    else:
        print("\nDo you approve this fix? (Y/N): ", end="", flush=True)
        user_input = input().strip().lower()
        decision = "approve" if user_input in {"y", "yes"} else "reject"

    return {
        "hitl_gate": "pre_healing",
        "hitl_decision": decision,
    }


def route_after_pre_healing_hitl(state: QAState) -> str:
    if state.get("hitl_decision") == "approve":
        return "healing_agent"
    else:
        # Reject or limit reached: skip healing, go straight to report
        return "report_agent"


# ============================================================================
# HEALINGAGENT — Diagnose failures, propose fix
# ============================================================================

def healing_agent(state: QAState) -> QAState:
    """Rule-based diagnosis of test execution failures.
    
    Phase 1 Strategy: Pattern-match on common failure modes in pytest output:
    - Selector/locator errors (xpath, css selector not found)
    - Timeout/wait issues (element didn't appear within timeout)
    - Assertion failures (assertion did not pass)
    - Navigation errors (page load failed, timeout navigating)
    
    Proposes simple, rule-based fixes based on the detected pattern.
    
    Phase 2 Upgrade: Replace with LLM-based diagnosis via prompt:
    "Given this pytest failure log, what's the root cause and what fix would you propose?"
    LLM can then do deeper reasoning about the failure mode and suggest more nuanced fixes.
    
    Phase 2 will also enable HealingAgent to actually apply fixes (locator updates,
    wait additions, etc.) and re-run ExecutionAgent automatically for tighter loops.
    """
    print("\n[HealingAgent] Diagnosing test failure (Phase 1: rule-based)...")
    
    execution_log = state.get("execution_log", "")
    healing_attempts = state.get("healing_attempts", 0) + 1  # Increment on entry
    
    # Rule-based pattern matching on common failure modes
    healing_proposal = _diagnose_failure_pattern(execution_log)
    
    print(f"Diagnosis: {healing_proposal}")
    return {
        "healing_proposed": healing_proposal,
        "healing_applied": False,  # Phase 1 only proposes; doesn't apply
        "healing_attempts": healing_attempts,
    }


def _diagnose_failure_pattern(log: str) -> str:
    """Rule-based diagnosis: match failure patterns in pytest output.
    
    Returns a plain-English diagnosis + proposed fix.
    """
    log_lower = log.lower()
    
    # Selector/locator failures
    if any(pat in log_lower for pat in ["elementnotfounderror", "not found", "no such element", "timeout finding element"]):
        if "xpath" in log_lower:
            return "Selector Issue (XPath): The XPath selector no longer matches any element on the page. " \
                   "Proposed Fix: Review the XPath in the BDD/generated script; update it to match the current DOM structure. " \
                   "Check for dynamic class names, ID changes, or nested element restructuring."
        elif "css" in log_lower or "selector" in log_lower:
            return "Selector Issue (CSS): The CSS selector no longer matches any element on the page. " \
                   "Proposed Fix: Update the CSS selector to reflect current page structure. " \
                   "Consider using more stable selectors (e.g., data-testid instead of class names)."
        else:
            return "Element Not Found: A required element was not found on the page. " \
                   "Proposed Fix: Verify the page loaded correctly. Review and update selectors. " \
                   "Ensure elements are visible/not hidden by CSS (display: none, visibility: hidden)."
    
    # Timeout/wait issues
    if any(pat in log_lower for pat in ["timeout", "timed out", "waiting for", "wait_for"]):
        return "Timing/Wait Issue: An element or action took longer than expected to complete. " \
               "Proposed Fix: Increase wait timeouts in the test script. " \
               "Add explicit waits (e.g., page.wait_for_selector()) before interacting with elements. " \
               "Check for slow network or blocking JavaScript."
    
    # Assertion failures
    if any(pat in log_lower for pat in ["assertion", "assert ", "failed", "expected", "actual"]):
        return "Assertion Failed: Test assertion did not pass. " \
               "Proposed Fix: Verify expected value is correct. " \
               "Check if the application behavior or UI text changed. " \
               "Review the BDD case for unclear or incorrect expectations."
    
    # Navigation/page load failures
    if any(pat in log_lower for pat in ["navigation", "navigate", "goto", "load", "404", "network"]):
        return "Navigation/Page Load Issue: Failed to navigate to or load the target page. " \
               "Proposed Fix: Verify the URL is correct and the target app is running. " \
               "Check network connectivity and firewall rules. " \
               "Ensure the page doesn't have authentication redirects in the test environment."
    
    # Generic fallback
    return "Unknown Failure: Could not automatically diagnose the failure from the logs. " \
           "Proposed Fix: Review the full pytest output above. " \
           "Check if the test script syntax is valid (imports, async/await, fixture usage). " \
           "Verify the page/app being tested is accessible and behaves as expected."


def route_after_healing_agent(state: QAState) -> str:
    # After HealingAgent proposes, route to HITL gate for approval
    return "pre_healing_hitl"


# ============================================================================
# REPORTAGENT — Aggregate results, request final approval
# ============================================================================

def assemble_report(state: QAState) -> QAState:
    """Assemble final report from all artifacts."""
    # Simple markdown assembly for Phase 1
    report = f"""# QA Report

## Summary
- **Run ID:** {state.get('run_id', 'unknown')}
- **Test Type:** {state.get('test_type', 'unknown')}
- **Execution Result:** {state.get('execution_result', 'not run')}

## User Story
{state.get('user_story', '')}

## Analysis
{state.get('analysis', '')}

## BDD Test Cases
{state.get('bdd_cases', '')}

## Review Notes
{state.get('review_notes', '')}

## Execution Log
{state.get('execution_log', '')}

## Healing Actions
{state.get('healing_proposed', 'None')}

---
Generated at {datetime.now().isoformat()}
"""
    return {"final_report": report}


def report_agent(state: QAState) -> QAState:
    """Wrap assemble_report."""
    return assemble_report(state)


# ============================================================================
# PUBLISH HITL GATE — Before final report is saved/published
# ============================================================================

def pre_publish_hitl(state: QAState) -> QAState:
    """HITL Gate 3: Approve before publishing report?"""
    print("\n" + "="*70)
    print("HITL GATE 3: Pre-Publish Approval")
    print("="*70)
    print(f"\nReport Preview (first 500 chars):")
    print(state.get('final_report', '')[:500])
    if AUTO_APPROVE:
        print("\nAuto-approve enabled: approving publication.")
        decision = "approve"
    else:
        print("\nDo you approve publication? (Y/N): ", end="", flush=True)
        user_input = input().strip().lower()
        decision = "approve" if user_input in {"y", "yes"} else "reject"

    return {
        "hitl_gate": "pre_publish",
        "hitl_decision": decision,
    }


def route_after_pre_publish_hitl(state: QAState) -> str:
    if state.get("hitl_decision") == "approve":
        return "save_report"
    else:
        # Reject: loop back to supervisor for revision
        return "supervisor"


# ============================================================================
# SAVE REPORT — Final output
# ============================================================================

def save_report(state: QAState) -> QAState:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(state.get("final_report", ""), encoding="utf-8")
    print(f"\n✓ Report saved to {OUTPUT_FILE}")
    return {"output_path": str(OUTPUT_FILE)}


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph(checkpointer=None):
    """Build the Phase 1 supervisor + swarm topology.
    
    Supervisor at entry, all other agents peer-to-peer swarm handoffs.
    HITL gates interrupt at 3 points: pre-execution, pre-healing, pre-publish.
    """
    graph = StateGraph(QAState)
    
    # Add all nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("blocked_report", blocked_report)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("analyze_story", analyze_story)
    graph.add_node("write_dynamic_memory", write_dynamic_memory)
    graph.add_node("generate_bdd", generate_bdd)
    graph.add_node("validate_bdd", validate_bdd)
    graph.add_node("test_case_review_agent", test_case_review_agent)
    graph.add_node("pre_execution_hitl", pre_execution_hitl)
    graph.add_node("execution_agent", execution_agent)
    graph.add_node("pre_healing_hitl", pre_healing_hitl)
    graph.add_node("healing_agent", healing_agent)
    graph.add_node("report_agent", report_agent)
    graph.add_node("pre_publish_hitl", pre_publish_hitl)
    graph.add_node("save_report", save_report)
    
    # Set entry point to Supervisor
    graph.set_entry_point("supervisor")
    
    # Supervisor routing
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"blocked_report": "blocked_report", "retrieve_memory": "retrieve_memory"},
    )
    graph.add_edge("blocked_report", "pre_publish_hitl")
    
    # TestCaseGen chain: retrieve → analyze → dynamic_memory → generate → validate
    graph.add_edge("retrieve_memory", "analyze_story")
    graph.add_edge("analyze_story", "write_dynamic_memory")
    graph.add_edge("write_dynamic_memory", "generate_bdd")
    graph.add_edge("generate_bdd", "validate_bdd")
    graph.add_conditional_edges(
        "validate_bdd",
        route_after_validate_bdd,
        {"generate_bdd": "generate_bdd", "test_case_review_agent": "test_case_review_agent"},
    )
    
    # TestCaseReview → pre-execution HITL
    graph.add_conditional_edges(
        "test_case_review_agent",
        route_after_review,
        {"pre_execution_hitl": "pre_execution_hitl"},
    )
    
    # Pre-execution HITL → execution or report
    graph.add_conditional_edges(
        "pre_execution_hitl",
        route_after_pre_execution_hitl,
        {"execution_agent": "execution_agent", "report_agent": "report_agent"},
    )
    
    # Execution → pre-healing HITL or report
    graph.add_conditional_edges(
        "execution_agent",
        route_after_execution,
        {"pre_healing_hitl": "pre_healing_hitl", "report_agent": "report_agent"},
    )
    
    # Pre-healing HITL → healing or report
    graph.add_conditional_edges(
        "pre_healing_hitl",
        route_after_pre_healing_hitl,
        {"healing_agent": "healing_agent", "report_agent": "report_agent"},
    )
    
    # Healing → back to pre-healing HITL (for re-approval after agent's proposal)
    graph.add_edge("healing_agent", "pre_healing_hitl")
    
    # Report → pre-publish HITL
    graph.add_edge("report_agent", "pre_publish_hitl")
    
    # Pre-publish HITL → save or back to supervisor
    graph.add_conditional_edges(
        "pre_publish_hitl",
        route_after_pre_publish_hitl,
        {"save_report": "save_report", "supervisor": "supervisor"},
    )
    
    # Save report → END
    graph.add_edge("save_report", END)
    
    return graph.compile(checkpointer=checkpointer)


# ============================================================================
# ENTRY POINTS
# ============================================================================

@traceable(name="run_full_qa_workflow_phase1")
def run_workflow(user_story: str) -> QAState:
    """Single entry point for one full Phase 1 workflow run."""
    app = build_graph()
    return cast(QAState, app.invoke({"user_story": user_story}))


def run_multiturn(user_story: str, thread_id: str, checkpointer=None) -> QAState:
    """Run with multi-turn memory (same thread_id resumes state)."""
    checkpointer = checkpointer or MemorySaver()
    app = build_graph(checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return cast(QAState, app.invoke({"user_story": user_story}, config=config))


# ============================================================================
# BATCH RUNNER — CI/CD-style: classify + generate + cache scripts for all
# (or one) user story, without interactive execution/healing/HITL.
#
# This is the "test generation" stage of a CI pipeline: given natural-
# language user stories, produce classified, reviewed BDD cases and cached
# pytest-playwright scripts. The full execute/heal/publish loop (with HITL)
# remains a separate, interactive concern -- run via --demo for one story
# at a time, since HITL approval doesn't make sense unattended across 13
# stories in one pass.
# ============================================================================

def generate_script_for_story(story_id: str, test_type: str, bdd_cases: str) -> tuple[Path, bool]:
    """Generate (or reuse cached) pytest-playwright script for a story.

    Caching rule (design doc Section 8): smoke and regression are cached --
    if a script already exists for this story, reuse it instead of
    regenerating (stable, repeated flows). Sanity and exploratory always
    regenerate fresh (novel/one-off by nature), but the result is still
    saved locally for every type, per design decision.

    Script filenames are deterministic (test_<story_id>.py, not run_id-based)
    specifically so this cache check can find them across separate runs.

    A cached file only counts as valid if it has real pytest content -- an
    empty or truncated file (e.g. left over from a run cut short by a rate
    limit or other failure) is treated as a cache-miss and regenerated,
    rather than being silently reused forever.

    Returns (script_path, was_cached).
    """
    tests_dir = PROJECT_DIR / "tests" / "phase1"
    tests_dir.mkdir(parents=True, exist_ok=True)
    script_path = tests_dir / f"test_{story_id}.py"

    if test_type in ("smoke", "regression") and _is_valid_script(script_path):
        print(f"  [{story_id}] {test_type}: cached script exists, reusing {script_path.name}")
        return script_path, True

    print(f"  [{story_id}] {test_type}: generating script...")
    script_code = _generate_playwright_script(bdd_cases, node_name=f"batch_codegen_{story_id}")

    if not _is_valid_script_content(script_code):
        raise RuntimeError(
            f"LLM returned an empty or invalid script for story '{story_id}' "
            f"({len(script_code.strip())} chars, 'def test_' present: {'def test_' in script_code}). "
            f"Not saving -- likely an LLM/API failure (e.g. rate limit) rather than a real script."
        )

    script_path.write_text(script_code, encoding="utf-8")
    print(f"  [{story_id}] saved {script_path.name}")
    return script_path, False


def _is_valid_script_content(content: str) -> bool:
    """A generated script only counts as valid if it has real pytest content."""
    return len(content.strip()) > 50 and "def test_" in content


def _is_valid_script(path: Path) -> bool:
    """Same check as _is_valid_script_content, but for a file already on disk --
    used by the cache check so an empty/broken cached file isn't reused forever."""
    if not path.exists():
        return False
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return _is_valid_script_content(content)


# ============================================================================
# PHASE 2a — DOM-grounded script generation (design doc Section 9.1)
#
# Problem: Phase 1's _generate_playwright_script has no knowledge of the real
# target page -- the LLM invents plausible-sounding selectors/nav text/class
# names for "a typical portfolio site" and is usually wrong. scan_target_page
# fixes this by inspecting the ACTUAL rendered page once, then feeding those
# real facts into the codegen prompt instead of letting the model guess.
# ============================================================================

def scan_target_page(url: str = TARGET_APP_URL, timeout_ms: int = 30000) -> dict:
    """Fetch the real target page and extract structural facts to ground
    codegen: actual nav link text, headings, image alt-text state, external
    links present, and a sample of real CSS classes in use.

    Uses Playwright's sync API directly (a one-off inspection, not a pytest
    test) -- run once per batch, not once per story, since every story in
    this project targets the same single-page site.
    """
    from playwright.sync_api import sync_playwright

    facts: dict = {
        "nav_links": [], "headings": [], "images_total": 0,
        "images_missing_alt": 0, "external_links": [], "classes_sample": [],
    }
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)

        nav_texts = page.locator("nav a, header a").all_inner_texts()
        facts["nav_links"] = [t.strip() for t in nav_texts if t.strip()][:15]

        heading_texts = page.locator("h1, h2, h3").all_inner_texts()
        facts["headings"] = [t.strip() for t in heading_texts if t.strip()][:15]

        images = page.locator("img").all()
        facts["images_total"] = len(images)
        for img in images:
            alt = img.get_attribute("alt")
            if alt is None:  # attribute genuinely absent -- a real violation.
                # Note: alt="" (present but empty) is NOT counted here -- that's the
                # correct, deliberate WCAG pattern for decorative images, not a bug.
                # Counting it as "missing" would produce false positives on sites that
                # correctly mark decorative images this way.
                facts["images_missing_alt"] += 1

        links = page.locator("a[href^='http']").all()
        hrefs = set()
        for link in links:
            href = link.get_attribute("href")
            if href and url not in href:
                hrefs.add(href)
        facts["external_links"] = sorted(hrefs)[:20]

        try:
            facts["classes_sample"] = page.eval_on_selector_all(
                "[class]",
                "els => [...new Set(els.map(e => e.className).filter(c => typeof c === 'string' && c.trim()))].slice(0, 40)",
            )
        except Exception:
            facts["classes_sample"] = []

        browser.close()
    return facts


def _format_page_facts(facts: dict) -> str:
    """Render scanned page facts into a compact block for the codegen prompt.

    Nav links are rendered as a literal Python list (not comma-joined prose) --
    an earlier version joined them as text and the model mis-transcribed the
    order/count when copying it into an assertion. Giving it as a literal
    list the model can paste directly removed that failure mode.
    """
    nav_list_literal = repr(facts["nav_links"])
    return (
        f"Navigation link text actually found, in DOM order -- use this EXACT list,\n"
        f"do not reorder, drop, or add entries: {nav_list_literal}\n"
        f"Headings actually found: {', '.join(facts['headings']) or '(none found)'}\n"
        f"Images: {facts['images_total']} total, {facts['images_missing_alt']} missing alt text\n"
        f"External links actually found on the page: {', '.join(facts['external_links'][:10]) or '(none found)'}\n"
        f"Sample of real CSS classes present on the page: {', '.join(facts['classes_sample']) or '(none found)'}"
    )


def _generate_grounded_playwright_script(bdd_cases: str, page_facts: dict, node_name: str) -> str:
    """Phase 2a codegen: same contract as _generate_playwright_script, but the
    prompt is grounded in a real DOM scan instead of letting the model guess.

    Two lessons folded in from the first real test of this prompt (see design
    doc Section 9.1.2 for the full before/after analysis):
    - Bot-domain status-code handling is given as literal, copy-pasteable
      code rather than a prose rule -- prose instructions are not reliably
      followed for a single specific edge case buried in a longer prompt.
    - The model has hallucinated a nonexistent Playwright method
      (`Locator.all_attribute_values`) and matched "Work" against "Frameworks"
      via unqualified has_text substring matching -- both are now called out
      explicitly as things to avoid.
    """
    facts_summary = _format_page_facts(page_facts)
    prompt = f"""Convert the following BDD Gherkin test cases into a pytest-playwright Python script.

The target application is at: {TARGET_APP_URL}

REAL PAGE STRUCTURE (scanned directly from the live page just now -- use this,
do NOT invent selectors, nav text, or class names that aren't listed here):
{facts_summary}

For checking external link status codes, use exactly this pattern (copy it as-is,
do not paraphrase or simplify the bot-domain handling):
```
BOT_BLOCKING_DOMAINS = ("linkedin.com",)

def is_healthy_status(url: str, status: int) -> bool:
    if any(domain in url for domain in BOT_BLOCKING_DOMAINS):
        return status in (200, 999)  # 999 = intentional anti-bot response, not broken
    return status == 200
```
Use `is_healthy_status(url, status)` in every external-link assertion instead of a bare
`status == 200` check.

Playwright API correctness -- these are real, verified Playwright sync API methods:
- To get an attribute from every element matched by a locator, loop over `locator.all()`
  and call `.get_attribute("href")` on each element individually. Do NOT use
  `locator.all_attribute_values(...)` -- this method does not exist in Playwright.
- When filtering a locator by visible text with `has_text=`, always pass `exact=True`
  unless you specifically want substring matching (e.g. `page.get_by_role("link", name="Work", exact=True)`).
  Without `exact=True`, "Work" will also match "Frameworks" as a substring.
- To assert a count is greater than zero (not an exact count), get the count as a plain
  Python integer with `locator.count()` and use a normal `assert count > 0`. Do NOT invent
  assertion-helper method names like `to_have_count_greater_than` -- if you are not
  certain a Playwright assertion method exists exactly as named, use `.count()` plus a
  plain assert instead of guessing at a fluent-assertion method name.
- Elements can have an `aria-label` that differs from their visible text (e.g. a logo
  link showing "SA" visually may have aria-label="Shalini Agarwal home"). The REAL PAGE
  STRUCTURE above lists visible text only. If a `get_by_role(..., name=...)` lookup for
  a nav/header element seems uncertain, prefer a CSS/text-content locator (e.g.
  `page.locator("header").get_by_text("Shalini Agarwal")`) over guessing the exact
  accessible name string.
- When checking images for alt text, the correct accessibility check is whether the
  `alt` ATTRIBUTE EXISTS (`img.get_attribute("alt") is not None`), not whether it is
  non-empty. `alt=""` (present but empty) is the correct, deliberate WCAG pattern for
  purely decorative images and must PASS. Only a genuinely MISSING alt attribute
  (`get_attribute("alt")` returns `None`) is a real violation. Do not flag `alt=""` as
  a failure -- that produces a false positive on sites that correctly mark decorative
  images this way.
- Most Playwright action/wait methods (`wait_for_load_state`, `goto`, `click`, `fill`,
  `wait_for_selector`, etc.) return `None` on success. They signal failure by RAISING
  an exception (e.g. a timeout error), not by returning a falsy value. NEVER wrap these
  in `assert method_call(...)` -- `assert page.wait_for_load_state(...)` will fail even
  on success, since `None` is falsy. Call these as plain statements; if the condition
  isn't met in time, Playwright raises automatically and pytest already treats an
  uncaught exception as a failure -- no extra assert is needed or correct. Only wrap a
  call in `assert` when its documented return type is genuinely a boolean (e.g.
  `locator.is_visible()`, `locator.is_checked()`).

The script MUST:
- Use the SYNCHRONOUS pytest-playwright API (NOT async/await, NOT playwright.async_api)
- Use the built-in `page` fixture provided automatically by the pytest-playwright plugin
- Follow pytest naming conventions (test_* functions), using plain `def`, never `async def`
- Only reference navigation text, headings, class names, or links that appear in the
  REAL PAGE STRUCTURE above. If a BDD scenario references something not confirmed to
  exist on the page (e.g. a hypothetical "deleted repo" link or a "certificate" link
  not seen above), write that test as a general robustness/negative check (e.g. "no
  broken links found", "no console errors") rather than asserting a specific unverified
  element is present
- Use `from playwright.sync_api import expect` for assertions where appropriate
- Include a short docstring on each test function mapping it back to its BDD scenario
- Be ready to run as-is with: pytest <script_name> -v

BDD Cases:
{bdd_cases}

Generate ONLY the complete Python script -- no explanations, no markdown code fences."""
    raw = _llm_text(get_llm(), prompt, node_name=node_name)
    return _strip_code_fences(raw)


def generate_grounded_script_for_story(story_id: str, test_type: str, bdd_cases: str, page_facts: dict) -> tuple[Path, bool]:
    """Phase 2a equivalent of generate_script_for_story -- same caching rule
    (smoke/regression reuse a valid cached script; sanity/exploratory always
    regenerate), but writes to tests/phase2/ and uses the DOM-grounded prompt.
    """
    tests_dir = PROJECT_DIR / "tests" / "phase2"
    tests_dir.mkdir(parents=True, exist_ok=True)
    script_path = tests_dir / f"test_{story_id}.py"

    if test_type in ("smoke", "regression") and _is_valid_script(script_path):
        print(f"  [{story_id}] {test_type}: cached (phase2) script exists, reusing {script_path.name}")
        return script_path, True

    print(f"  [{story_id}] {test_type}: generating DOM-grounded script...")
    script_code = _generate_grounded_playwright_script(bdd_cases, page_facts, node_name=f"phase2_codegen_{story_id}")

    if not _is_valid_script_content(script_code):
        raise RuntimeError(
            f"LLM returned an empty or invalid phase2 script for story '{story_id}' "
            f"({len(script_code.strip())} chars, 'def test_' present: {'def test_' in script_code})."
        )

    script_path.write_text(script_code, encoding="utf-8")
    print(f"  [{story_id}] saved {script_path.name}")
    return script_path, False


def process_single_story(user_story_text: str, story_id: str) -> QAState:
    """Run one story through Supervisor -> TestCaseGenAgent -> TestCaseReviewAgent.

    Deliberately stops before ExecutionAgent/HITL -- this is the batch
    generation+classification pass. Runs node functions directly in sequence
    rather than via build_graph(), since batch mode doesn't need graph-level
    branching beyond the BDD retry loop (handled here with a plain for-loop).
    """
    state: QAState = {"user_story": user_story_text}
    state.update(load_static_memory(state))
    state.update(supervisor(state))

    if state.get("input_blocked"):
        return state

    state.update(retrieve_memory(state))
    state.update(analyze_story(state))
    state.update(write_dynamic_memory(state))

    for _ in range(MAX_BDD_RETRIES + 1):
        state.update(generate_bdd(state))
        state.update(validate_bdd(state))
        if state.get("bdd_valid"):
            break

    state.update(test_case_review_agent(state))
    return state


def run_batch(story_filter: str | None = None, seed: bool = False, phase: str = "phase1") -> None:
    """CI/CD-style batch entry point.

    Args:
        story_filter: if given, process only this story id. Default: all
                       stories found under user_stories/.
        seed: if True, run the setup step first (scripts/generate_user_stories.py
              + seed_vector_db.py) before processing -- generates fresh user
              stories from data/portfolio_content.json and re-indexes the
              vector DB. Off by default so a normal batch run doesn't
              regenerate stories/embeddings every time.
        phase: "phase1" (default) uses the original ungrounded codegen.
               "phase2" scans the real target page once, then uses the
               DOM-grounded codegen for every story -- see design doc
               Section 9.1. Scripts/logs are written to tests/<phase>/ and
               outputs/test_results/<phase>/ respectively, so both phases'
               output sit side by side for the same story id.
    """
    if seed:
        print("Running setup: generating user stories + seeding vector DB...\n")
        subprocess.run([sys.executable, "scripts/generate_user_stories.py"], cwd=PROJECT_DIR, check=True)
        subprocess.run([sys.executable, "seed_vector_db.py"], cwd=PROJECT_DIR, check=True)
        print("\nSetup complete.\n")

    story_files = sorted((PROJECT_DIR / "user_stories").glob("**/*.md"))
    if not story_files:
        raise SystemExit("No user stories found under user_stories/. Run with --seed first.")

    if story_filter:
        story_files = [
            p for p in story_files
            if parse_story_file(p)[0].get("id", p.stem) == story_filter
        ]
        if not story_files:
            raise SystemExit(f"Story id '{story_filter}' not found under user_stories/.")

    page_facts = None
    if phase == "phase2":
        print(f"Scanning real target page ({TARGET_APP_URL}) once for all stories in this run...\n")
        try:
            page_facts = scan_target_page(TARGET_APP_URL)
            print(f"Scan complete: {len(page_facts['nav_links'])} nav links, "
                  f"{page_facts['images_total']} images ({page_facts['images_missing_alt']} missing alt), "
                  f"{len(page_facts['external_links'])} external links found.\n")
        except Exception as e:
            raise SystemExit(
                f"Failed to scan {TARGET_APP_URL} for phase2 grounding: {e}\n"
                f"Ensure Playwright browsers are installed (playwright install chromium) "
                f"and the target URL is reachable."
            )

    results = []
    for path in story_files:
        metadata, body = parse_story_file(path)
        story_id = metadata.get("id", path.stem)
        print(f"\n{'='*70}\nProcessing: {story_id} ({phase})\n{'='*70}")

        try:
            state = process_single_story(body, story_id)

            if state.get("input_blocked"):
                print(f"  [{story_id}] BLOCKED: {state.get('block_reason')}")
                results.append({
                    "story_id": story_id, "test_type": "-", "status": "blocked",
                    "script": "-", "cached": False, "execution_result": "-", "log": "-",
                })
                continue

            test_type = state.get("test_type", "sanity")
            if phase == "phase2":
                script_path, cached = generate_grounded_script_for_story(
                    story_id, test_type, state.get("bdd_cases", ""), page_facts
                )
            else:
                script_path, cached = generate_script_for_story(story_id, test_type, state.get("bdd_cases", ""))

            print(f"  [{story_id}] running pytest...")
            execution_result, _, log_path = _run_pytest_script(script_path, log_name=story_id, phase=phase)

            results.append({
                "story_id": story_id,
                "test_type": test_type,
                "status": "ok",
                "script": str(script_path.relative_to(PROJECT_DIR)),
                "cached": cached,
                "execution_result": execution_result,
                "log": str(log_path.relative_to(PROJECT_DIR)),
            })
        except Exception as e:
            error_text = str(e)
            print(f"  [{story_id}] ERROR: {error_text}")
            if "rate_limit" in error_text.lower() or "429" in error_text:
                print(
                    f"  [{story_id}] This looks like an API rate limit, not a code bug. "
                    f"Batch runs make many LLM calls per story (~5-6 x {len(story_files)} stories) -- "
                    f"a hosted free-tier daily token cap can be exhausted well before 13 stories finish. "
                    f"Consider switching back to a local Ollama model for batch runs (no external quota), "
                    f"or wait for the quota to reset and re-run -- the cache-validity check means already-"
                    f"succeeded stories won't be wastefully regenerated."
                )
            results.append({
                "story_id": story_id, "test_type": "-", "status": "error",
                "script": "-", "cached": False, "execution_result": "error",
                "log": error_text[:100],
            })
            continue

    print(f"\n\n{'='*70}\nBATCH SUMMARY ({phase})\n{'='*70}")
    print(f"{'Story ID':<22} {'Type':<12} {'Result':<8} {'Script (cached?)':<40} Log")
    print("-"*70)
    for r in results:
        note = " (cached)" if r["cached"] else ""
        script_col = f"{r['script']}{note}"
        print(f"{r['story_id']:<22} {r['test_type']:<12} {r['execution_result']:<8} {script_col:<40} {r['log']}")
    passed = sum(1 for r in results if r["execution_result"] == "pass")
    errored = sum(1 for r in results if r["execution_result"] == "error")
    print(f"\nTotal: {len(results)} stories processed. {passed} passed, {errored} errored, {len(results) - passed - errored} failed/blocked.")
    print(f"Full pytest logs saved under: {(OUTPUTS_DIR / 'test_results' / phase).relative_to(PROJECT_DIR)}/")


def batch_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 1 QA workflow.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the original single-story interactive demo (full pipeline incl. execution/healing/publish HITL) instead of batch mode.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve all HITL gates. Only applies in --demo mode.",
    )
    parser.add_argument(
        "--user-story",
        default="As a user, I want to reset my password so I can regain access if I forget it.",
        help="User story to process. Only applies in --demo mode.",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Batch mode: run setup (generate user stories + seed vector DB) before processing.",
    )
    parser.add_argument(
        "--story",
        default=None,
        help="Batch mode: process only this story id (default: all stories under user_stories/).",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Batch mode: use Phase 2a DOM-grounded codegen (scans the real target page first) "
             "instead of Phase 1's ungrounded codegen. Writes to tests/phase2/ and "
             "outputs/test_results/phase2/ so both phases' output are directly comparable.",
    )
    args = parser.parse_args(argv)

    global AUTO_APPROVE
    AUTO_APPROVE = args.auto_approve or AUTO_APPROVE

    if args.demo:
        result = run_workflow(args.user_story)
        print(f"\n✓ Workflow complete. Report: {result.get('output_path')}")
    else:
        run_batch(story_filter=args.story, seed=args.seed, phase="phase2" if args.phase2 else "phase1")


if __name__ == "__main__":
    batch_main()
