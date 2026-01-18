import streamlit as st

st.set_page_config(page_title="QTGuard", layout="wide")
st.title("QTGuard — Offline QT Medication Safety Copilot")
st.caption(
    "Research/demo prototype for decision support. Not a medical device."
    "Do not use for autonomous clinical decisions."
)

st.markdown(
    "Paste a de-identified mini-chart (meds + QTc + labs + risk factors).\n"
    "This scaffold will be connected to MedGemma inference next."
)

mini_chart = st.text_area(
    "Mini-chart input",
    height=220,
    placeholder="Example: QTc=520 ms; K=3.1; Mg=1.6; Meds: ...",
)

if st.button("Generate plan"):
    st.info("Model not wired yet. Next step: connect MedGemma inference and JSON-structured outputs.")
    st.json(
        {
            "risk_summary": "Placeholder",
            "action_plan": ["Placeholder"],
            "patient_counseling": "Placeholder",
            "audit_view": {"missing_data": [], "notes": ["Placeholder"]},
        }
    )
