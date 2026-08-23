# Revenue Reasoning Agent — POC

7-capability agentic system: leakage detection, forecasting, anomaly detection,
RCA, what-if, recommendations, segmentation. Planner/Critic reasoning loop over
a model registry; every decision logged to an append-only Reasoning Ledger.

## Layout
```
core/       state.py (RunState + ReasoningLedger), llm.py (Ollama client)
stages/     ingest.py -> eda_profiler.py -> feature_engine.py -> feature_analysis.py
capabilities/  (Night day-block: forecasting, anomaly, leakage, ...)
domain_profiles/  (utilities / insurance / banking YAML vocab + rules)
ui/         (Streamlit app — day block)
tests/      make_demo_data.py, test_night1.py
```

## Setup
```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b && ollama pull gemma2:2b && ollama pull deepseek-r1:7b
python tests/test_night1.py    # end-to-end smoke test, no LLM needed
```

Stages 0-1.6 are fully deterministic — test passes without Ollama running.


## Running the UI

```
pip install streamlit
streamlit run app.py
```

1. Upload a CSV (try `demo_utilities.csv` or `hardmode_utilities.csv` from the tests generator)
2. Type what you want in plain language
3. Approve the plan card (human-in-the-loop)
4. Watch the reasoning ledger stream, then explore the tabs
5. Export the report + raw ledger from the Report tab

Ollama toggle: auto-detected. Off = deterministic fallbacks (fully functional).
