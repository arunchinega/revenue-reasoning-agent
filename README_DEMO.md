# Revenue Reasoning Agent (RRA) — POC

Agentic revenue analysis: upload billing data, ask in plain language; the
agent profiles, plans, races 9 forecasting models, detects anomalies (5-detector
consensus), quantifies revenue leakage against ground truth, explains root
cause, and signs every decision in an append-only reasoning ledger.

**Reasoning patterns:** CoT in every LLM prompt · ReAct in the Planner→Executor→Critic
loop · ToT over the model bake-off (branch → evaluate → prune/blend) ·
number-guarded hybrid narration · human-in-the-loop plan approval.

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Demo data: `python tests/make_external_data.py` (emits CSV + ground-truth manifest).

## Tests (14)
`python tests/test_full_run.py` etc — all deterministic, no Ollama needed.
LLM mode: install Ollama + `llama3.2:3b` (fast) / `llama3.1:8b` (quality) / `gemma2:2b` (narrator).
