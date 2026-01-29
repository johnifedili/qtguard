import os
import sys
import json
import streamlit as st

# Ensure repo root is on sys.path when running `streamlit run app/streamlit_app.py`
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qtguard_core.rag_pipeline import run_qtguard_with_retrieval

ASSETS_PATH = os.path.join(ROOT_DIR, "assets", "outputs.jsonl")


def load_demo_outputs(path: str) -> dict:
    demos = {}
    if not os.path.exists(path):
        return demos
    with open(path, "r", encoding="utf-8") as f:
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


def render_evidence_panel(evidence):
    st.subheader("Evidence (retrieved + reranked)")
    if not evidence:
        st.warning("No evidence retrieved.")
        return

    for i, e in enumerate(evidence, start=1):
        with st.expander(
            f"[E{i}] {e.title} — {e.section} (score={e.score:.3f})",
            expanded=(i == 1),
        ):
            st.write(e.text)
            st.caption(f"chunk_id: {e.chunk_id}")


# Load demos (mini-charts only; output will NOT come from DEMOS anymore)
DEMOS = load_demo_outputs(ASSETS_PATH)

demo_labels = {
    "high_risk_polypharmacy": "High-risk polypharmacy (Mini-chart)",
    "missing_data_deferral": "Missing data deferral (Mini-chart)",
}

available_demo_keys = [k for k in demo_labels.keys() if k in DEMOS]

# UI
st.set_page_config(page_title="QTGuard", layout="wide")
st.title("QTGuard — Offline QT Medication Safety Copilot")
st.caption(
    "Research/demo prototype for decision support. Not a medical device. "
    "Do not use for autonomous clinical decisions."
)

st.markdown(
    "**This version generates output via retrieval-driven logic** (BM25 + vector retrieval + reranking + safety gating). "
    "Demo cases only provide sample mini-charts; output is computed live from the retriever."
)

with st.sidebar:
    st.markdown("### Retrieval settings")
    score_threshold = st.slider(
        "Evidence score threshold",
        0.0, 5.0, 1.5, 0.1,
        help="If the top evidence score is below this, QTGuard will recommend safe deferral.",
    )
    st.caption("Tip: if you see frequent deferrals, lower the threshold slightly and/or expand your KB sources.")

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
    # ALWAYS generate via retrieval-driven pipeline (demo and custom)
    out_dict, evidence, weak = run_qtguard_with_retrieval(
        mini_chart,
        score_threshold=score_threshold,
    )

    render_evidence_panel(evidence)
    st.divider()

    st.subheader("Risk summary")
    st.write(out_dict.get("risk_summary", ""))

    st.subheader("Action plan")
    for i, item in enumerate(out_dict.get("action_plan", []), start=1):
        st.write(f"{i}. {item}")

    st.subheader("Patient-friendly counseling")
    st.write(out_dict.get("patient_counseling", ""))

    st.subheader("Audit view")
    render_audit_view(out_dict.get("audit_view", {}))

