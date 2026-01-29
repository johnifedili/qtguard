from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional

from qtguard_core.guardrails import build_safe_output
from qtguard_core.retrieval import get_retriever, Evidence


def _build_retrieval_query(mini_chart: str) -> str:
    # Include key phrases we want to retrieve reliably (electrolyte targets + monitoring)
    return (
        "QT prolongation torsades QTc telemetry repeat ECG "
        "potassium 4.0 5.0 mEq/L K repletion target magnesium Mg. "
        + mini_chart
    )


def _evidence_notes(evidence: List[Evidence], top_n: int = 5) -> List[str]:
    notes: List[str] = []
    if not evidence:
        notes.append("No evidence retrieved.")
        return notes
    notes.append("Evidence retrieved (reranked):")
    for i, e in enumerate(evidence[:top_n], start=1):
        notes.append(f"[E{i}] {e.title} — {e.section} — score={e.score:.3f} — {e.chunk_id}")
    return notes


def _strip_noise_notes(notes: List[str]) -> List[str]:
    """
    Remove legacy model/inference noise so audit stays retrieval-focused.
    Prevents gated-model errors from appearing in retrieval mode.
    """
    bad_substrings = [
        "medgemma",
        "gated repo",
        "cannot access gated repo",
        "401 client error",
        "you are trying to access a gated repo",
        "please log in",
    ]
    cleaned: List[str] = []
    for n in notes:
        nl = n.lower()
        if any(b in nl for b in bad_substrings):
            continue
        cleaned.append(n)
    return cleaned


def _is_evidence_weak(
    evidence: List[Evidence],
    score_threshold: float,
    margin_threshold: float = 0.5,
) -> Tuple[bool, float, float]:
    """
    Cross-encoder scores are raw logits and may be negative.
    Weak evidence if:
      - no evidence, OR
      - top score < score_threshold, OR
      - top score not clearly better than #2 (low margin)
    """
    if not evidence:
        return True, float("-inf"), 0.0

    top = evidence[0].score
    second = evidence[1].score if len(evidence) > 1 else float("-inf")
    margin = (top - second) if second != float("-inf") else float("inf")
    weak = (top < score_threshold) or (margin < margin_threshold)
    return weak, top, margin


def _extract_float(pattern: str, text: str) -> Optional[float]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(float(m.group(1)))
    except Exception:
        return None


def _extract_meds(text: str) -> List[str]:
    m = re.search(r"meds?\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if not m:
        return []
    meds_blob = m.group(1)
    meds = [x.strip() for x in re.split(r",|;|\n", meds_blob) if x.strip()]
    meds = [re.sub(r"\s+PRN.*$", "", x, flags=re.IGNORECASE).strip() for x in meds]
    return meds


def _build_evidence_guided_plan(mini_chart: str) -> Tuple[str, List[str], str]:
    """
    Conservative, evidence-guided plan builder.
    Uses mini-chart signals (QTc/K/Mg/HR + meds list) and phrases actions cautiously.
    """
    qtc = _extract_int(r"qtc\s*=\s*([0-9.]+)", mini_chart)
    hr = _extract_int(r"hr\s*=\s*([0-9.]+)", mini_chart)
    k = _extract_float(r"\bk\s*=\s*([0-9.]+)", mini_chart)
    mg = _extract_float(r"\bmg\s*=\s*([0-9.]+)", mini_chart)
    meds = _extract_meds
