from unittest.mock import Mock

from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


def test_health_contract():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_run_workflow_returns_stable_response_shape(monkeypatch):
    monkeypatch.setattr(api, "run_workflow", Mock(return_value={
        "analysis": "analysis", "bdd_cases": "Given a user", "bdd_valid": True,
        "retry_count": 0, "final_report": "report", "output_path": "out.md",
    }))
    response = client.post("/api/run-workflow", json={"user_story": "As a user, I can sign in."})
    assert response.status_code == 200
    assert response.json()["analysis"] == "analysis"
    assert response.json()["input_blocked"] is False
    api.run_workflow.assert_called_once_with("As a user, I can sign in.")


def test_check_guardrails_returns_input_and_output_results(monkeypatch):
    monkeypatch.setattr(api, "screen_user_story", Mock(return_value=Mock(passed=True, reason="ok")))
    monkeypatch.setattr(api, "check_bdd_output_schema", Mock(return_value=Mock(passed=False, reason="missing Then")))
    response = client.post("/api/check-guardrails", json={"user_story": "A safe story", "bdd_cases": "bad"})
    assert response.status_code == 200
    assert response.json() == {
        "user_story": {"passed": True, "reason": "ok"},
        "bdd_cases": {"passed": False, "reason": "missing Then"},
    }


def test_rag_eval_contract_and_top_k(monkeypatch):
    monkeypatch.setattr(api, "run_rag_evaluation", Mock(return_value=[
        Mock(hit=True, query="q1", expected_section="projects", found_sections=["projects"]),
        Mock(hit=False, query="q2", expected_section="skills", found_sections=[]),
    ]))
    response = client.post("/api/rag-eval", json={"top_k": 2})
    assert response.status_code == 200
    assert response.json()["hit_rate"] == 0.5
    assert response.json()["total"] == 2
    api.run_rag_evaluation.assert_called_once_with(top_k=2)


def test_rag_eval_rejects_out_of_range_top_k():
    assert client.post("/api/rag-eval", json={"top_k": 0}).status_code == 422
    assert client.post("/api/rag-eval", json={"top_k": 11}).status_code == 422


def test_observability_empty_log_contract(monkeypatch):
    monkeypatch.setattr(api, "read_log", Mock(return_value=[]))
    response = client.get("/api/observability")
    assert response.status_code == 200
    assert response.json() == {"total_calls": 0, "by_node": []}
