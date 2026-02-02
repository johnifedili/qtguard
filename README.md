
[![CI](https://github.com/johnifedili/qtguard/actions/workflows/ci.yml/badge.svg)](https://github.com/johnifedili/qtguard/actions/workflows/ci.yml)


# QTGuard — Offline QT Medication Safety Copilot (MedGemma / HAI-DEF)

QTGuard is a human-centered, offline-capable medication safety copilot that helps clinicians (especially inpatient/ED pharmacists) triage **QT-prolongation risk** from a compact, de-identified “mini-chart” and produce a structured, actionable plan.

This project is built for the **MedGemma Impact Challenge** (Kaggle) and uses **MedGemma (HAI-DEF)** as the local reasoning engine.

## What it does (demo output)
Given a mini-chart (med list + QTc + key labs + risk factors), QTGuard generates:

1. **Risk summary** (drivers and severity framing)
2. **Action plan** (stop/switch candidates, monitoring cadence, electrolyte correction prompts, escalation triggers)
3. **Patient-friendly counseling** (plain language explanation)
4. **Audit view** (inputs used + missing data + safe deferrals)


**Try it:** paste a mini-chart → click **Generate plan** → inspect Evidence + Audit view for traceability.

> Disclaimer: QTGuard is a research/demo prototype for decision support. It is not a medical device and must not be used for autonomous clinical decisions.

---

## Quickstart (local)

1) Create a virtual environment:
- `python -m venv .venv`
- `source .venv/bin/activate`  # macOS/Linux
- `.venv\Scripts\activate`     # Windows (PowerShell)

2) Install dependencies:
- `pip install -U pip`
- `pip install -r requirements.txt`

3) Run the Streamlit demo (recommended):
- `STREAMLIT_SERVER_FILE_WATCHER_TYPE=none python -m streamlit run app/streamlit_app.py`

Open in your browser:
- `http://127.0.0.1:8501` (preferred)
- `http://localhost:8501`


## Offline evaluation harness

QTGuard includes an offline evaluation harness that measures:
- **Deferral accuracy** (safe deferral when key inputs are missing)
- **Evidence keyword recall** (retrieval surfaces expected mitigation concepts)
- **Plan keyword recall** (generated plan includes expected actions)
- **Composite score** (quick iteration metric)

Run the eval suite:
- `python scripts/eval.py`

Saved artifacts (written per run):
- `reports/eval_runs/<YYYYMMDD_HHMMSS>/summary.json`
- `reports/eval_runs/<YYYYMMDD_HHMMSS>/per_case.jsonl`

Optional: keep eval reports out of git by adding this line to `.gitignore`:
- `reports/`


