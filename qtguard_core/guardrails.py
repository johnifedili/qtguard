from __future__ import annotations

import re
from typing import List, Tuple

from qtguard_core.schema import QTGuardOutput


_REQUIRED_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("QTc", re.compile(r"\bqtc\b|\bqt\s*c\b|\bqt interval\b", re.IGNORECASE)),
    ("Potassium (K)", re.compile(r"\bK\b|\bpotassium\b", re.IGNORECASE)),
    ("Magnesium (Mg)", re.compile(r"\bMg\b|\bmagnesium\b", re.IGNORECASE)),
]

# Optional but helpful
_OPTIONAL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Heart rate (HR)", re.compile(r"\bHR\b|\bheart rate\b", re.IGNORECASE)),
]


def find_missing_inputs(mini_chart: str) -> List[str]:
    """Very lightweight missing-data detector for the demo scaffold."""
    text = mini_chart or ""
    missing = []
    for label, pattern in _REQUIRED_PATTERNS:
        if not pattern.search(text):
            missing.append(label)
    return missing


def build_safe_output(mini_chart: str) -> QTGuardOutput:
    """
    If missing required fields, return deferral output.
    Otherwise return placeholder (until MedGemma inference is wired).
    """
    missing = find_missing_inputs(mini_chart)
    if missing:
        return QTGuardOutput.deferral(missing)

    # Placeholder "safe" output while inference is not yet connected.
    return QTGuardOutput(
        risk_summary=(
            "Inputs appear present. Model inference is not wired yet, so this is a placeholder summary."
        ),
        action_plan=[
            "Placeholder: connect MedGemma inference to generate a structured plan.",
            "Placeholder: include monitoring cadence and escalation triggers.",
            "Placeholder: include medication review and electrolyte management prompts.",
        ],
        patient_counseling=(
            "Placeholder counseling: this tool is a demo and does not replace clinician judgment."
        ),
        audit_view={
            "missing_data": [],
            "notes": [
                "Guardrails check passed for required inputs (QTc, K, Mg).",
                "Returning placeholder output until MedGemma inference is connected.",
            ],
        },
    )
