from src.workflow import build_graph
def test_graph_runs():
    app = build_graph()
    state = {"user_story": "As a user, I want to reset my password."}
    result = app.invoke(state)
    assert "final_report" in result
