"""Build evidence packets and record executive release decisions.

This records an approver-supplied decision; it does not create or imply approval.
The JSON digest detects later edits, but is not a cryptographic identity signature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def latest_git_revision() -> str | None:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def blockers_for_report(report: dict[str, Any], report_id: str) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    status = report.get("status")
    if status and status != "PASS":
        blockers.append({"id": f"{report_id}:overall-status", "description": f"Evaluation status is {status}"})
    thresholds = report.get("thresholds")
    if isinstance(thresholds, dict):
        for name, outcome in thresholds.items():
            if isinstance(outcome, dict) and outcome.get("status") not in ("PASS", None):
                blockers.append({"id": f"{report_id}:threshold:{name}", "description": f"{name}: {outcome.get('status')} ({outcome.get('value')})"})
    for suite_name, suite in report.get("suites", {}).items() if isinstance(report.get("suites"), dict) else []:
        if not isinstance(suite, dict):
            continue
        for outcome in suite.get("thresholds", []):
            if isinstance(outcome, dict) and not outcome.get("ok", False):
                blockers.append({"id": f"{report_id}:{suite_name}:{outcome.get('metric')}", "description": f"{suite_name} {outcome.get('metric')}: {outcome.get('value')} (threshold {outcome.get('threshold')})"})
        aggregate = report.get("status")
        if aggregate == "FAIL" and not suite.get("thresholds"):
            blockers.append({"id": f"{report_id}:{suite_name}:status", "description": f"Suite {suite_name} has failed status"})
    metrics = report.get("aggregate", {})
    if isinstance(metrics, dict):
        pending = metrics.get("manual_review_pending", 0)
        failed = metrics.get("manual_review_failed", 0)
        if pending:
            blockers.append({"id": f"{report_id}:human-review-pending", "description": f"{pending} human review(s) pending"})
        if failed:
            blockers.append({"id": f"{report_id}:human-review-failed", "description": f"{failed} human review(s) failed"})
    return blockers


def capture_coverage(api_url: str, output: Path) -> dict[str, Any]:
    url = api_url.rstrip("/") + "/api/observability"
    headers = {}
    token = os.getenv("OBSERVABILITY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    coverage = payload.get("trace_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("API response does not include trace_coverage")
    trace_ids = coverage.get("request_trace_ids", [])
    coverage["ingestion_verified_at"] = datetime.now(timezone.utc).isoformat()
    if not trace_ids:
        coverage["langsmith_ingestion_verified"] = False
        coverage["ingestion_verification_error"] = "No local LangSmith request span ids were available"
    else:
        import os
        from langsmith import Client
        api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            raise ValueError("LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is required to verify trace ingestion")
        client = Client(api_key=api_key)
        found: set[str] = set()
        for offset in range(0, len(trace_ids), 100):
            batch = trace_ids[offset:offset + 100]
            found.update(str(run.id) for run in client.list_runs(run_ids=batch, project_name=coverage.get("project"), select=["id"], limit=len(batch)))
        coverage["ingested_trace_ids"] = len(found)
        coverage["langsmith_ingestion_verified"] = len(found) == len(set(trace_ids)) == coverage.get("total_requests")
        coverage["ingestion_verification_error"] = None if coverage["langsmith_ingestion_verified"] else "One or more request spans were not found in LangSmith"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    return coverage


def build_packet(release_id: str, report_paths: list[Path], coverage_path: Path, output_root: Path) -> Path:
    if not ID_RE.fullmatch(release_id):
        raise ValueError("release id must contain 1-80 letters, digits, dots, underscores, or hyphens")
    if not report_paths:
        raise ValueError("Provide at least one --evaluation report")
    reports = []
    blockers = []
    for path in report_paths:
        report = read_json(path)
        report_id = str(report.get("run_id") or report.get("dataset_name") or path.stem)
        findings = []
        for case in report.get("results", []):
            review = case.get("human_review") if isinstance(case, dict) else None
            verdict = review.get("verdict") if isinstance(review, dict) else None
            if isinstance(case, dict) and (case.get("risk") == "high" or not case.get("automatic_checks_passed", True) or verdict in ("pending", "fail")):
                findings.append({"case_id": case.get("case_id"), "risk": case.get("risk"),
                                 "automatic_checks_passed": case.get("automatic_checks_passed"),
                                 "human_review": review})
        reports.append({"id": report_id, "path": str(path.resolve()), "sha256": sha256(path),
                        "created_at": report.get("created_at"), "status": report.get("status", "UNKNOWN"),
                        "system_variants": report.get("system_variants", {}),
                        "aggregate": report.get("aggregate", {}), "thresholds": report.get("thresholds", {}),
                        "suites": report.get("suites", {}), "risk_findings": findings})
        blockers.extend(blockers_for_report(report, report_id))
    coverage = read_json(coverage_path)
    total = int(coverage.get("total_requests", 0))
    configured = int(coverage.get("trace_configured_requests", 0))
    ratio = coverage.get("trace_configured_coverage")
    if coverage.get("release_id") != release_id:
        blockers.append({"id": "production:release-mismatch", "description": f"Trace snapshot release {coverage.get('release_id')} does not match packet release {release_id}"})
    if not coverage.get("required"):
        blockers.append({"id": "production:environment", "description": "Trace evidence was not captured in production mode"})
    if not coverage.get("configured"):
        blockers.append({"id": "production:langsmith-config", "description": "LangSmith tracing is not configured/enabled"})
    if total < 1:
        blockers.append({"id": "production:no-traffic", "description": "No retained production API requests were measured"})
    if ratio is None or ratio < 1.0 or configured != total:
        blockers.append({"id": "production:trace-coverage", "description": f"Trace-configured request coverage is {ratio}; 100% required"})
    if coverage.get("langsmith_ingestion_verified") is not True:
        blockers.append({"id": "production:ingestion-unverified", "description": "LangSmith backend ingestion has not been independently verified"})
    packet = {
        "release_id": release_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": latest_git_revision(),
        "evaluation_reports": reports,
        "production_trace_coverage": coverage,
        "blockers": blockers,
        "decision": "PENDING_EXECUTIVE_REVIEW",
    }
    target = output_root / release_id
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "decision_packet.json"
    md_path = target / "decision_packet.md"
    if (target / "executive_signoff.json").exists():
        raise ValueError("A sign-off already exists for this release; do not rewrite its evidence packet")
    json_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"# Executive release decision packet: {release_id}", "", f"- Created: {packet['created_at']}",
             f"- Source revision: {packet['source_revision'] or 'unavailable'}", "", "## Evaluation evidence", ""]
    for report in reports:
        lines.append(f"- **{report['id']}** â€” {report['status']} â€” SHA-256 `{report['sha256']}` ({report['path']})")
    lines += ["", "## Production trace coverage", "",
              f"- Environment: {coverage.get('environment')}; release: {coverage.get('release_id')}",
              f"- Requests: {total}; trace-configured: {configured}; coverage: {ratio}",
              f"- LangSmith ingestion independently verified: {coverage.get('langsmith_ingestion_verified')}", "",
              "## Open blockers", ""]
    lines += [f"- `{item['id']}` â€” {item['description']}" for item in blockers] or ["- None"]
    lines += ["", "## Decision", "", "Pending executive review. A GO decision requires every blocker to be explicitly accepted with a control action.",
              "This packet does not constitute approval.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path


def record_signoff(packet_path: Path, approver_name: str, approver_title: str, decision: str,
                   rationale: str, accepted: list[str], controls: list[str]) -> Path:
    packet = read_json(packet_path)
    if not approver_name.strip() or not approver_title.strip() or not rationale.strip():
        raise ValueError("Approver name, title, and decision rationale are required")
    blockers = packet.get("blockers", [])
    blocker_ids = {item["id"] for item in blockers}
    unknown = sorted(set(accepted) - blocker_ids)
    if unknown:
        raise ValueError(f"Accepted blocker id(s) are not in this packet: {', '.join(unknown)}")
    if decision == "go":
        unaccepted = sorted(blocker_ids - set(accepted))
        if unaccepted:
            raise ValueError("GO is blocked until every open item is resolved or explicitly accepted: " + ", ".join(unaccepted))
        if accepted and not controls:
            raise ValueError("A GO decision accepting residual risks requires --control-action for every accepted blocker")
    parsed_controls = []
    for item in controls:
        if "=>" not in item:
            raise ValueError("Each --control-action must use BLOCKER_ID=>action text")
        blocker_id, action = item.split("=>", 1)
        if blocker_id not in set(accepted) or not action.strip():
            raise ValueError("Each control action must reference an accepted blocker and include an action")
        parsed_controls.append({"blocker_id": blocker_id, "action": action.strip()})
    if decision == "go" and set(accepted) - {item["blocker_id"] for item in parsed_controls}:
        raise ValueError("Every accepted blocker needs its own control action")
    record = {
        "release_id": packet["release_id"], "decision": decision.upper(),
        "approver": {"name": approver_name.strip(), "title": approver_title.strip()},
        "rationale": rationale.strip(), "accepted_blockers": sorted(set(accepted)),
        "control_actions": parsed_controls,
        "decision_at": datetime.now(timezone.utc).isoformat(),
        "packet_path": str(packet_path.resolve()), "packet_sha256": sha256(packet_path),
        "source_revision": packet.get("source_revision"),
    }
    target = packet_path.parent / "executive_signoff.json"
    if target.exists():
        raise ValueError(f"Sign-off already exists; preserve it and create a new packet: {target}")
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(canonical).hexdigest()
    target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-coverage", help="Fetch the production API trace-coverage snapshot")
    capture.add_argument("--api-url", required=True)
    capture.add_argument("--output", type=Path, required=True)
    packet = sub.add_parser("packet", help="Build a release evidence packet")
    packet.add_argument("--release-id", required=True)
    packet.add_argument("--evaluation", type=Path, action="append", required=True)
    packet.add_argument("--trace-coverage", type=Path, required=True)
    packet.add_argument("--output-root", type=Path, default=ROOT / "release_signoffs")
    sign = sub.add_parser("sign", help="Record an executive GO or NO-GO")
    sign.add_argument("--packet", type=Path, required=True)
    sign.add_argument("--approver-name", required=True)
    sign.add_argument("--approver-title", required=True)
    sign.add_argument("--decision", choices=["go", "no-go"], required=True)
    sign.add_argument("--rationale", required=True)
    sign.add_argument("--accept-blocker", action="append", default=[])
    sign.add_argument("--control-action", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.command == "capture-coverage":
            coverage = capture_coverage(args.api_url, args.output)
            print(f"Captured {coverage.get('total_requests')} requests; trace coverage={coverage.get('trace_configured_coverage')}")
        elif args.command == "packet":
            result = build_packet(args.release_id, args.evaluation, args.trace_coverage, args.output_root)
            print(f"Decision packet: {result}")
        else:
            result = record_signoff(args.packet, args.approver_name, args.approver_title, args.decision,
                                    args.rationale, args.accept_blocker, args.control_action)
            print(f"Executive decision recorded: {result}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
