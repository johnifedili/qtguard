import os
import sys
import json
import re
import streamlit as st

# Ensure repo root is on sys.path when running `streamlit run app/streamlit_app.py`
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qtguard_core.rag_pipeline import run_qtguard_with_retrieval
from qtguard_core.guardrails import build_safe_output

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
        title = getattr(e, "title", "Evidence")
        section = getattr(e, "section", "")
        score = getattr(e, "score", None)
        chunk_id = getattr(e, "chunk_id", None)

        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        with st.expander(
            f"[E{i}] {title} — {section} (score={score_str})",
            expanded=(i == 1),
        ):
            st.write(getattr(e, "text", ""))
            if chunk_id is not None:
                st.caption(f"chunk_id: {chunk_id}")


def _extract_missing_from_text(mini_chart: str) -> list[str]:
    """
    Detect explicit missing markers in the mini-chart (e.g., QTc=unknown, Mg: n/a).
    This is UI-level sanity so audit_view doesn't contradict safe deferral outputs.
    """
    t = (mini_chart or "").lower()
    missing = []
    tokens = r"(unknown|n\/a|na|none|null|pending|tbd)"

    if re.search(r"(qtc|qt\s*c|qt interval)\s*[:=]\s*" + tokens + r"\b", t):
        missing.append("QTc")
    if re.search(r"(potassium|\bk)\s*[:=]\s*" + tokens + r"\b", t):
        missing.append("Potassium (K)")
    if re.search(r"(magnesium|\bmg)\s*[:=]\s*" + tokens + r"\b", t):
        missing.append("Magnesium (Mg)")

    return missing


def _audit_fix_missing(out_dict: dict, mini_chart: str) -> dict:
    """
    Ensure audit_view.missing_data + notes are consistent with deferral language and explicit unknown values.
    This does NOT change model behavior; it only fixes audit metadata shown in the UI.
    Additionally, when QTc is missing, rewrite risk_summary to avoid misleading "high-risk signals present" language.
    """
    audit = out_dict.get("audit_view") or {}
    missing = audit.get("missing_data") or []
    missing_set = set(missing)

    # 1) If the action_plan contains a safe deferral message, extract missing labels from it
    plan_text = "\n".join(out_dict.get("action_plan", []) or []).lower()
    if "missing key inputs" in plan_text:
        # crude but effective: pull items after the colon
        # e.g., "Safe deferral: Missing key inputs: QTc, K."
        m = re.search(r"missing key inputs\s*:\s*([a-z0-9,\-\s\(\)]+)", plan_text)
        if m:
            raw = m.group(1)
            for token in [x.strip() for x in raw.split(",")]:
                if not token:
                    continue
                if "qtc" in token:
                    missing_set.add("QTc")
                if "potassium" in token or token in {"k"}:
                    missing_set.add("Potassium (K)")
                if "magnesium" in token or token in {"mg"}:
                    missing_set.add("Magnesium (Mg)")

    # 2) If mini-chart explicitly has unknown/pending values, treat them as missing
    for lab in _extract_missing_from_text(mini_chart):
        missing_set.add(lab)

    missing = sorted(missing_set)
    audit["missing_data"] = missing

    # If QTc is missing, make risk_summary clinically honest (avoid "higher risk signals present")
    if "QTc" in missing:
        meds_match = re.search(r"(?i)\bmeds?\s*:\s*(.+)", mini_chart or "")
        meds = meds_match.group(1).strip() if meds_match else ""
        if meds:
            meds_short = meds[:160] + ("..." if len(meds) > 160 else "")
            out_dict["risk_summary"] = (
                "QT risk cannot be fully assessed without QTc. "
                f"Medications listed include: {meds_short}. "
                "Obtain ECG/QTc to complete risk assessment."
            )
        else:
            out_dict["risk_summary"] = (
                "QT risk cannot be fully assessed without QTc. "
                "Obtain ECG/QTc to complete risk assessment."
            )

    # 3) Fix notes so they don't contradict missing_data
    notes = audit.get("notes") or []
    notes = [n for n in notes if "Guardrails check passed" not in n]

    if missing:
        notes.insert(0, f"Guardrails: missing required inputs: {', '.join(missing)}.")
    else:
        notes.insert(0, "Guardrails check passed for required inputs (QTc, K, Mg).")

    audit["notes"] = notes
    out_dict["audit_view"] = audit
    return out_dict


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
    "Demo mode loads precomputed MedGemma outputs (from `assets/outputs.jsonl`) for reliability in recordings.\n\n"
    "**New:** Enable retrieval-driven output to generate plans using retrieved evidence + safety gating.\n\n"
    "Note: cross-encoder relevance scores can be negative; tune thresholds accordingly."
)

with st.sidebar:
    st.markdown("### Output mode")
    use_retrieval_output = st.checkbox(
        "Use retrieval-driven output (recommended)",
        value=True,
        help="If enabled, QTGuard output is generated from retrieval + safety gating. "
             "If disabled, demo cases show precomputed outputs and custom inputs use guardrails.",
        key="use_retrieval_output_v1",
    )

    st.markdown("### Retrieval settings")
    score_threshold = st.slider(
        "Evidence score threshold",
        -10.0, 10.0, 0.0, 0.1,
        key="score_threshold_v2",
        help="Cross-encoder relevance scores are raw logits and may be negative. "
             "If the top score is below this, QTGuard will recommend safe deferral.",
    )

    st.caption("Debug")
    st.write(f"use_retrieval_output = {use_retrieval_output}")
    st.write(f"score_threshold = {score_threshold:.1f}")

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

# Persist last result so the page doesn't go "blank" on reruns
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if st.button("Generate plan"):
    mini_chart_clean = (mini_chart or "").strip()
    if not mini_chart_clean:
        st.warning("Please paste a mini-chart first (or select a demo case).")
        st.stop()

    with st.spinner("Generating..."):
        try:
            if use_retrieval_output:
                # Retrieval-driven pipeline
                out_dict, evidence, weak = run_qtguard_with_retrieval(
                    mini_chart_clean,
                    score_threshold=score_threshold,
                )

                # UI-level audit consistency fix (prevents missing-data contradictions)
                out_dict = _audit_fix_missing(out_dict, mini_chart_clean)

                st.session_state["last_result"] = {
                    "mode": "retrieval",
                    "out": out_dict,
                    "evidence": evidence,
                    "weak": weak,
                    "selected_case_id": selected_case_id,
                }
            else:
                # Original behavior (demo JSON or guardrails)
                if selected_case_id and selected_case_id in DEMOS:
                    out = DEMOS[selected_case_id]["output"]
                    st.session_state["last_result"] = {
                        "mode": "demo",
                        "out": out,
                        "evidence": None,
                        "weak": None,
                        "selected_case_id": selected_case_id,
                    }
                else:
                    output = build_safe_output(mini_chart_clean).model_dump()
                    st.session_state["last_result"] = {
                        "mode": "guardrails",
                        "out": output,
                        "evidence": None,
                        "weak": None,
                        "selected_case_id": None,
                    }
        except Exception as e:
            st.session_state["last_result"] = None
            st.error("Generate failed. See error details below.")
            st.exception(e)

# Render the last result (prevents blank screen + makes reruns stable)
if st.session_state["last_result"] is not None:
    result = st.session_state["last_result"]
    out_dict = result["out"]
    evidence = result.get("evidence")

    if result.get("mode") == "retrieval":
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
