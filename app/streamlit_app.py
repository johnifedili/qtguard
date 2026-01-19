import os
import sys

# Force repo root onto sys.path so `qtguard_core` imports work when running from /app
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from qtguard_core.guardrails import build_safe_output

st.set_page_config(page_title="QTGuard", layout="wide")
st.title("QTGuard — Offline QT Medication Safety Copilot")
st.caption(
    "Research/demo prototype for decision support. Not a medical device. "
    "Do not use for autonomous clinical decisions."
)

st.markdown(
    "Paste a de-identified mini-chart (meds + QTc + labs + risk factors).\n"
    "This demo uses guardrails + safe deferrals. Next step: connect MedGemma inference."
)

mini_chart = st.text_area(
    "Mini-chart input",
    height=240,
    placeholder="Example: QTc=520 ms; K=3.1; Mg=1.6; Meds: ...",
)

if st.button("Generate plan"):
    output = build_safe_output(mini_chart)

    st.subheader("Risk summary")
    st.write(output.risk_summary)

    st.subheader("Action plan")
    for i, item in enumerate(output.action_plan, start=1):
        st.write(f"{i}. {item}")

    st.subheader("Patient-friendly counseling")
    st.write(output.patient_counseling)

    st.subheader("Audit view")
    st.json(output.model_dump())

