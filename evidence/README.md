# Evidence guide

This directory contains the real outputs captured from the four lab steps.
Screenshots must be taken from the LangSmith dashboard or terminal after a
successful run; no synthetic traces or scores are used.

## Files

- `01_langsmith_traces.png`: LangSmith project showing at least 50 Step 1 runs.
- `01_langsmith_run_log.txt`: terminal log of the Step 1 run that produced those traces.
- `02_prompt_hub.png`: both pushed prompt names visible in Prompt Hub.
- `02_ab_routing_log.txt`: 50 deterministic request IDs with V1/V2 labels.
- `03_ragas_scores.png`: terminal comparison table for both prompt versions.
- `03_ragas_report.json`: copy of the tracked `data/ragas_report.json`.
- `04_pii_demo_log.txt`: PII redaction cases, including a clean case.
- `04_json_demo_log.txt`: valid, repaired, and unrecoverable JSON cases.

## V1 versus V2 analysis

After the RAGAS run, compare the four numeric fields in
`03_ragas_report.json`. V1 is optimized for concise answers, while V2 is
optimized for structured explanations. The final submission should state
which version has the higher faithfulness and why the retrieved context and
answer format account for the difference.
