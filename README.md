# QTGuard — Offline QT Medication Safety Copilot (MedGemma / HAI-DEF)

QTGuard is a human-centered, offline-capable medication safety copilot that helps clinicians (especially inpatient/ED pharmacists) triage **QT-prolongation risk** from a compact, de-identified “mini-chart” and produce a structured, actionable plan.

This project is built for the **MedGemma Impact Challenge** (Kaggle) and uses **MedGemma (HAI-DEF)** as the local reasoning engine.

## What it does (demo output)
Given a mini-chart (med list + QTc + key labs + risk factors), QTGuard generates:

1. **Risk summary** (drivers and severity framing)
2. **Action plan** (stop/switch candidates, monitoring cadence, electrolyte correction prompts, escalation triggers)
3. **Patient-friendly counseling** (plain language explanation)
4. **Audit view** (inputs used + missing data + safe deferrals)

> Disclaimer: QTGuard is a research/demo prototype for decision support. It is not a medical device and must not be used for autonomous clinical decisions.

---

## Quickstart (local)
### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
```
