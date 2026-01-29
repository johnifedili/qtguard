import os
import sys
import json
import streamlit as st

# Ensure repo root is on sys.path when running `streamlit run app/streamlit_app.py`
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qtguard_core.guardrails import build_safe_output
from qtguard_core.retrieval import get_retriever

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


def build_retrieval_query(mini_chart: str) -> str:
    # Simple, robust first-pass query. You can later parse meds/QTc explicitly.
    return f"QT prolongation torsades risk triage guidance. {mini_chart}"


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


def evidence_is_weak(evidence, threshold: float = 1.5) -> bool:
    """
    Cross-encoder scores are not probabilities; they can be > 1.
    Threshold is a practical starting point—tune using your eval set.
    """
    if not evidence:
        return True
    return evidence[0].score < threshold


# Load demos
DEMOS = load_demo_outputs(ASSETS_PATH)

demo_labels = {
    "high_risk_polypharmacy": "High-risk polypharmacy (MedGemma output)",
    "missing_data_deferral": "Missing data deferral (guardrail override)",
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
    "Demo mode loads precomputed MedGemma outputs (from `assets/outputs.jsonl`) for reliability in recordings. "
    "Custom inputs fall back to guardrails, but still retrieve and display evidence."
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

# Optional: make threshold adjustable in UI (helps tuning for Kaggle)
with st.sidebar:
    st.markdown("### Retrieval settings")
    score_threshold = st.slider("Evidence score threshold", 0.0, 5.0, 1.5, 0.1)
    st.caption("If top evidence score is below this, QTGuard will recommend safe deferral.")

if st.button("Generate plan"):
    # --- Retrieval (always runs) ---
    try:
        retriever = get_retriever()
        query = build_retrieval_query(mini_chart)
        evidence = retriever.search(query)
    except Exception as e:
        evidence = []
        st.error(f"Evidence retrieval failed: {e}")

    render_evidence_panel(evidence)

    weak = evidence_is_weak(evidence, threshold=score_threshold)
    top_score = evidence[0].score if evidence else float("-inf")

    st.divider()

    # --- Output ---
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
        # Append evidence notes for transparency (judge trust)
        audit = out.get("audit_view", {}) or {}
        notes = audit.get("notes", []) or []
        if weak:
            notes.append("Evidence confidence low → safe deferral recommended.")
        else:
            notes.append(f"Top evidence score={top_score:.3f} (OK)")
        if evidence:
            notes.append("Evidence retrieved:")
            for i, e in enumerate(evidence[:5], start=1):
                notes.append(f"[E{i}] {e.title} — {e.section} — score={e.score:.3f} — {e.chunk_id}")
        audit["notes"] = notes
        render_audit_view(audit)

    # Custom mode: guardrails + evidence-aware safety gate
    else:
        output = build_safe_output(mini_chart)

        # If evidence is weak, add a strong deferral note (safety maturity)
        audit = output.model_dump().get("audit_view", {}) or {}
        notes = audit.get("notes", []) or []

        if weak:
            notes.append("Evidence confidence low → safe deferral recommended.")
        else:
            notes.append(f"Top evidence score={top_score:.3f} (OK)")

        # Always attach evidence references (judge trust)
        if evidence:
            notes.append("Evidence retrieved:")
            for i, e in enumerate(evidence[:5], start=1):
                notes.append(f"[E{i}] {e.title} — {e.section} — score={e.score:.3f} — {e.chunk_id}")

        audit["notes"] = notes

        st.subheader("Risk summary")
        st.write(output.risk_summary)

        st.subheader("Action plan")
        for i, item in enumerate(output.action_plan, start=1):
            st.write(f"{i}. {item}")

        st.subheader("Patient-friendly counseling")
        st.write(output.patient_counseling)

        st.subheader("Audit view")
        render_audit_view(audit)
