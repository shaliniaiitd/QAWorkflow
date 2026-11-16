from src import qa_workflow


def test_batch_main_defaults_to_human_gate_demo(monkeypatch):
    seen = {}

    def fake_run_workflow(user_story):
        seen["workflow"] = user_story
        return {"output_path": "demo-report.md"}

    def fake_run_batch(**kwargs):
        seen["batch"] = kwargs

    monkeypatch.setattr(qa_workflow, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(qa_workflow, "run_batch", fake_run_batch)

    qa_workflow.batch_main([])

    assert seen["workflow"] == "As a visitor, I want to browse the portfolio projects and their descriptions so I can evaluate the work."
    assert "batch" not in seen


def test_execution_agent_passes_pytest_output_to_healer(monkeypatch, tmp_path):
    script_path = tmp_path / "test_generated_run.py"
    script_path.write_text("def test_case():\n    assert True\n", encoding="utf-8")
    log_path = tmp_path / "run.log"
    log_path.write_text("AssertionError: expected 1, got 2\n", encoding="utf-8")
    monkeypatch.setattr(qa_workflow, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(qa_workflow, "_generate_playwright_script", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        qa_workflow,
        "_run_pytest_script",
        lambda *args, **kwargs: ("fail", "short summary", log_path),
    )

    result = qa_workflow.execution_agent({
        "run_id": "run",
        "execution_script_path": str(script_path),
    })

    assert "AssertionError: expected 1, got 2" in result["execution_log"]


def test_healing_agent_prepares_candidate_from_failure(monkeypatch, tmp_path):
    script_path = tmp_path / "test_generated_run.py"
    script_path.write_text("def test_case():\n    assert True\n", encoding="utf-8")
    candidate = "def test_case():\n    assert True\n    # reviewed healing candidate\n"
    seen = {}

    def fake_llm_text(_llm, prompt, node_name):
        seen["prompt"] = prompt
        seen["node_name"] = node_name
        return candidate

    monkeypatch.setattr(qa_workflow, "_llm_text", fake_llm_text)
    monkeypatch.setattr(qa_workflow, "get_llm", lambda: object())

    result = qa_workflow.healing_agent({
        "execution_log": "AssertionError: expected 1, got 2",
        "execution_script_path": str(script_path),
    })

    assert "AssertionError: expected 1, got 2" in seen["prompt"]
    assert seen["node_name"] == "healing_agent"
    assert result["healing_script"] == candidate.strip()
    assert result["healing_applied"] is False
    assert not list(tmp_path.glob("*healed*.py"))


def test_healing_agent_does_not_rewrite_script_for_github_pages_404(monkeypatch):
    def unexpected_llm_call(*args, **kwargs):
        raise AssertionError("healer should not ask the LLM to repair a missing target site")

    monkeypatch.setattr(qa_workflow, "_llm_text", unexpected_llm_call)
    result = qa_workflow.healing_agent({
        "execution_log": (
            'heading "404" [level=1]\n'
            "The site configured at this address does not contain the requested file."
        ),
    })

    assert "returned 404" in result["healing_proposed"]
    assert result["healing_script"] == ""
    assert result["healing_applied"] is False


def test_apply_healing_saves_approved_candidate_separately(monkeypatch, tmp_path):
    monkeypatch.setattr(qa_workflow, "PROJECT_DIR", tmp_path)
    original = tmp_path / "tests" / "phase1" / "test_generated_run.py"
    original.parent.mkdir(parents=True)
    original.write_text("def test_original():\n    assert False\n", encoding="utf-8")
    candidate = "def test_original():\n    assert True\n    # approved healing candidate\n"

    result = qa_workflow.apply_healing({
        "run_id": "run",
        "healing_attempts": 1,
        "healing_script": candidate,
    })

    healed_path = tmp_path / "tests" / "phase1" / "test_generated_run_healed_1.py"
    assert result["healing_applied"] is True
    assert result["execution_script_path"] == str(healed_path)
    assert healed_path.read_text(encoding="utf-8") == candidate
    assert original.read_text(encoding="utf-8") == "def test_original():\n    assert False\n"
