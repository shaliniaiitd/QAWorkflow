// File: frontend/src/api.js
// Thin fetch wrapper around api.py's endpoints. React (browser-side) has
// no way to call Python directly -- every action here is a real HTTP
// request, unlike Streamlit where the button click and the Python logic
// live in the same process.

const BASE_URL = "http://localhost:8000";

async function post(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export const runWorkflow = (userStory) => post("/api/run-workflow", { user_story: userStory });

export const checkGuardrails = (userStory, bddCases, useLlamaguard) =>
  post("/api/check-guardrails", { user_story: userStory, bdd_cases: bddCases, use_llamaguard: useLlamaguard });

export const runRagEval = (topK) => post("/api/rag-eval", { top_k: topK });

export const getObservability = () => get("/api/observability");
