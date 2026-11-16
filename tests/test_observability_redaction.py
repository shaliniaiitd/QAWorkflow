import pytest

from src import qa_workflow
from src.utils import observability
from src.utils.observability import _redact_trace_payload


def test_trace_payload_redacts_pii_recursively():
    payload = {
        "story": "Contact jane.doe@example.com or 9876543210; ID 1234 5678 9012",
        "nested": ["Keep this text", {"response": "mail x@y.com"}],
    }

    redacted = _redact_trace_payload(payload)

    assert "jane.doe@example.com" not in redacted["story"]
    assert "9876543210" not in redacted["story"]
    assert "1234 5678 9012" not in redacted["story"]
    assert redacted["nested"][0] == "Keep this text"
    assert redacted["nested"][1]["response"] == "mail [REDACTED_EMAIL]"


def test_traceable_registers_redaction_processors(monkeypatch):
    captured = {}

    def fake_traceable(*args, **kwargs):
        captured.update(kwargs)
        return lambda function: function

    monkeypatch.setattr(observability, "_langsmith_traceable", fake_traceable)

    @observability.traceable(name="redaction-test")
    def traced(value):
        return value

    assert callable(traced)
    assert captured["process_inputs"] is _redact_trace_payload
    assert captured["process_outputs"] is _redact_trace_payload


def test_pytest_failure_log_redacts_pii(monkeypatch, tmp_path):
    script_path = tmp_path / "test_pii_output.py"
    script_path.write_text(
        "def test_failure():\n"
        "    assert False, 'jane.doe@example.com 9876543210 1234 5678 9012'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(qa_workflow, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(qa_workflow, "OUTPUTS_DIR", tmp_path)

    result, _, log_path = qa_workflow._run_pytest_script(
        script_path, log_name="pii-redaction-test", phase="phase2"
    )

    log = log_path.read_text(encoding="utf-8")
    assert result == "fail"
    assert "jane.doe@example.com" not in log
    assert "9876543210" not in log
    assert "1234 5678 9012" not in log
    assert "[REDACTED_EMAIL]" in log
    assert "[REDACTED_PHONE]" in log
    assert "[REDACTED_ID]" in log


@pytest.mark.parametrize("module_path", ["src.qa_workflow", "src.workflow"])
def test_saved_report_redacts_pii(module_path, monkeypatch, tmp_path):
    import importlib

    workflow_module = importlib.import_module(module_path)
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(workflow_module, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(workflow_module, "OUTPUT_FILE", report_path)

    workflow_module.save_report({"final_report": "User jane.doe@example.com, phone 9876543210"})

    saved_report = report_path.read_text(encoding="utf-8")
    assert "jane.doe@example.com" not in saved_report
    assert "9876543210" not in saved_report
    assert "[REDACTED_EMAIL]" in saved_report
    assert "[REDACTED_PHONE]" in saved_report
