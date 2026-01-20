import os
import sys
import json
import streamlit as st

# Ensure repo root is on sys.path when running `streamlit run app/streamlit_app.py`
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qtguard_core.guardrails import build_safe_output

ASSETS_PATH = os.path.join(ROOT_DIR, "assets", "outputs.jsonl")


def load_demo_outputs(path: str) -> dict:
    demos = {}
    if not os.path.exists(path):
        return demos
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            demos[obj["case_id"]] = obj
    return demos


def render_audit_view(audit_view: dict):
    missing = audit_view.get("missing_data", [])
    notes = audit_view.get("notes", [])

    st.markdown("**Missing data**")
    if missing:
        for item in missing:
            st.write(f"- {item}")
    else:
        st.write("- None")

    st.markdown("**Notes**")
    if notes:
        for n in notes:
            st.write(f"- {n}")
    else:
        st.write("- None")


DEMOS = load_demo_outputs(ASSETS_PATH)

demo_labels = {
    "high_risk_polypharmacy": "High-risk polypharmacy (MedGemma output)",
    "missing_data_deferral": "Missing data deferral (guardrail override)",
}

available_demo_keys = [k for k in demo_labels.keys() if k in DEMOS]

st.set_page_config(page_title="QTGuard", layout="wide")
st.title("QTGuard — Offline QT Medication Safety Copilot")
st.caption(
    "Research/demo prototype for decision support. Not a medical device. "
    "Do not use for autonomous clinical decisions."
)

st.markdown(
    "Demo mode loads precomputed MedGemma outputs (from `assets/outputs.jsonl`) for reliability in recordings. "
    "Custom inputs fall back to guardrails."
)

selected_label = st.selectbox(
    "Select a demo case",
    options=["(Custom input)"] + [demo_labels[k] for k in available_demo_keys],
)

selected_case_id = None
default_text = ""
if selected_label != "(Custom input)":
    for k in available_demo_keys:
        if demo_labels[k] == selected_label:
            selected_case_id = k
            break
    default_text = DEMOS[selected_case_id]["mini_chart"]

mini_chart = st.text_area(
    "Mini-chart input",
    height=240,
    value=default_text,
    placeholder="Example: QTc=520 ms; K=3.1; Mg=1.6; Meds: ...",
)

if st.button("Generate plan"):
    # Demo mode: show precomputed output
    if selected_case_id and selected_case_id in DEMOS:
        out = DEMOS[selected_case_id]["output"]

        st.subheader("Risk summary")
        st.write(out.get("risk_summary", ""))

        st.subheader("Action plan")
        for i, item in enumerate(out.get("action_plan", []), start=1):
            st.write(f"{i}. {item}")

        st.subheader("Patient-friendly counseling")
        st.write(out.get("patient_counseling", ""))

        st.subheader("Audit view")
        render_audit_view(out.get("audit_view", {}))

    # Custom mode: fall back to guardrails
    else:
        output = build_safe_output(mini_chart)

        st.subheader("Risk summary")
        st.write(output.risk_summary)

        st.subheader("Action plan")
        for i, item in enumerate(output.action_plan, start=1):
            st.write(f"{i}. {item}")

        st.subheader("Patient-friendly counseling")
        st.write(output.patient_counseling)

        st.subheader("Audit view")
        render_audit_view(output.model_dump().get("audit_view", {}))

