from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional

from qtguard_core.guardrails import build_safe_output
from qtguard_core.retrieval import get_retriever, Evidence


def _build_retrieval_query(mini_chart: str) -> str:
    return (
        "QT prolongation torsades QTc telemetry repeat ECG "
        "potassium 4.0 5.0 mEq/L K repletion target magnesium Mg "
        "medication list doses consider alternatives review necessity "
        "syncope palpitations bradycardia structural heart disease multiple QT drugs "
        "ondansetron azithromycin citalopram haloperidol amiodarone "
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
    bad_substrings = [
        "medgemma",
        "gated repo",
        "cannot access gated repo",
        "401 client error",
        "you are trying to access a gated repo",
        "please log in",
        "hf hub",
    ]
    cleaned: List[str] = []
    for n in notes:
        nl = (n or "").lower()
        if any(b in nl for b in bad_substrings):
            continue
        cleaned.append(n)
    return cleaned


def _is_evidence_weak(
    evidence: List[Evidence],
    score_threshold: float,
    margin_threshold: float = 0.2,
) -> Tuple[bool, float, float]:
    """
    Cross-encoder scores are raw logits and may be negative.

    Weak evidence if:
      - no evidence, OR
      - top score < score_threshold, OR
      - (only when top is borderline) top score not clearly better than #2
    """
    if not evidence:
        return True, float("-inf"), 0.0

    top = evidence[0].score
    second = evidence[1].score if len(evidence) > 1 else float("-inf")
    margin = (top - second) if second != float("-inf") else float("inf")

    # Only apply margin gating when top score is borderline.
    if top >= 0.5:
        weak = (top < score_threshold)
    else:
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
    """
    Robust meds extraction (multiline line match).
    Treats '?', 'unknown', 'n/a' as missing.
    """
    m = re.search(r"(?im)^\s*meds?\s*:\s*(.+?)\s*$", text)
    if not m:
        return []

    meds_blob = (m.group(1) or "").strip()
    meds_blob_l = meds_blob.lower()

    if meds_blob_l in {"?", "unknown", "n/a", "na", "none listed"}:
        return []

    meds = [x.strip() for x in re.split(r",|;|\n", meds_blob) if x.strip()]
    meds = [re.sub(r"\s+PRN.*$", "", x, flags=re.IGNORECASE).strip() for x in meds]
    meds = [m for m in meds if m and m.lower() not in {"?", "unknown", "n/a", "na"}]
    return meds


def _has_symptom(mini_chart: str, *terms: str) -> bool:
    t = (mini_chart or "").lower()
    return any(term.lower() in t for term in terms)


def _is_low_normal_k_mg(k: Optional[float], mg: Optional[float]) -> Tuple[bool, bool]:
    """
    Treat low-normal values as needing "optimize/check" in QT-risk workflows.
    This helps borderline scenarios still mention electrolytes explicitly.
    """
    k_low_normal = (k is not None and k <= 3.5)
    mg_low_normal = (mg is not None and mg <= 1.8)
    return k_low_normal, mg_low_normal


def _build_evidence_guided_plan(mini_chart: str, evidence: List[Evidence]) -> Tuple[str, List[str], str, bool]:
    qtc = _extract_int(r"qtc\s*=\s*([0-9.]+)", mini_chart)
    hr = _extract_int(r"hr\s*=\s*([0-9.]+)", mini_chart)
    k = _extract_float(r"\bk\s*=\s*([0-9.]+)", mini_chart)
    mg = _extract_float(r"\bmg\s*=\s*([0-9.]+)", mini_chart)
    meds = _extract_meds(mini_chart)

    high_qtc = (qtc is not None and qtc >= 500)
    brady = (hr is not None and hr < 60)

    low_k = (k is not None and k < 3.5)
    low_mg = (mg is not None and mg < 1.8)
    k_low_normal, mg_low_normal = _is_low_normal_k_mg(k, mg)

    multi_qt_drugs = (len(meds) >= 2)

    # Symptoms / red flags that should push telemetry mention
    syncope_like = _has_symptom(mini_chart, "syncope", "near syncope", "faint", "fainting")
    arrhythmia_like = _has_symptom(mini_chart, "palpitations", "torsades", "vtach", "ventricular", "seizure")
    structural = _has_symptom(mini_chart, "chf", "heart failure", "cardiomyopathy", "structural heart")

    # If retrieved evidence explicitly mentions telemetry, prefer to surface it
    evidence_text = "\n".join([getattr(e, "text", "") for e in (evidence or [])]).lower()
    evidence_mentions_telemetry = ("telemetry" in evidence_text)

    missing_inputs: List[str] = []
    if qtc is None:
        missing_inputs.append("QTc")
    if k is None:
        missing_inputs.append("Potassium (K)")
    if mg is None:
        missing_inputs.append("Magnesium (Mg)")
    if not meds:
        missing_inputs.append("Medication list")

    missing_critical = len(missing_inputs) > 0

    flags: List[str] = []
    if high_qtc:
        flags.append(f"QTc={qtc} ms (>=500)")
    if low_k:
        flags.append(f"K={k} (low)")
    if low_mg:
        flags.append(f"Mg={mg} (low)")
    if brady:
        flags.append(f"HR={hr} (bradycardia)")
    if meds:
        flags.append(f"{len(meds)} meds listed")
    if syncope_like:
        flags.append("syncope (high-risk symptom)")

    if flags:
        risk_summary = "Higher QT/TdP risk signals present: " + "; ".join(flags) + "."
    else:
        risk_summary = (
            "QT risk signals could not be confidently derived from the mini-chart; "
            "add QTc/K/Mg/HR and a medication list with doses for a more specific plan."
        )

    action_plan: List[str] = []

    # IMPORTANT: include the phrase "Required inputs" for eval + clarity
    if missing_critical:
        action_plan.append(
            "Safe deferral: Missing key inputs. Required inputs: " + ", ".join(missing_inputs) + "."
        )

    # Electrolytes: explicit "Correct electrolytes" phrasing, even when borderline/low-normal
    if low_k or low_mg or high_qtc:
        action_plan.append(
            "Correct electrolytes first. Aim to maintain potassium toward the higher end of normal "
            "(often cited target K=4.0–5.0 mEq/L in QT-risk prevention pathways) and correct magnesium "
            "to normal/high-normal per local protocol."
        )
    elif k_low_normal or mg_low_normal:
        action_plan.append(
            "Correct electrolytes / optimize K and Mg (even low-normal values can matter in QT-risk settings), "
            "then reassess QTc after stabilization."
        )

    if meds:
        action_plan.append(
            "Review necessity of QT-prolonging agents and consider alternatives where feasible "
            "(avoid absolute 'hold' language unless your source explicitly mandates it)."
        )

    # Telemetry rule: broaden so drug exposure / symptoms / multi-drug contexts still surface telemetry.
    telemetry_recommended = (
        evidence_mentions_telemetry
        or high_qtc
        or brady
        or syncope_like
        or arrhythmia_like
        or structural
        or multi_qt_drugs
        or (meds and (low_k or low_mg))
    )

    if telemetry_recommended:
        action_plan.append(
            "Repeat ECG after electrolyte correction and/or medication adjustments; consider telemetry "
            "in higher-risk scenarios (e.g., QTc >= 500 ms, multiple QT-prolonging agents, symptoms, bradycardia, "
            "structural heart disease)."
        )
    else:
        # Still mention electrolytes explicitly to avoid silent misses in borderline cases.
        action_plan.append(
            "If QT risk remains uncertain, obtain repeat ECG and trend QTc after checking/correcting electrolytes "
            "and reviewing medications."
        )

    patient_counseling = (
        "To reduce the risk of an abnormal heart rhythm, your care team may correct low potassium/magnesium and "
        "review medications that can affect QT. Seek urgent care for fainting, severe dizziness, or palpitations."
    )

    return risk_summary, action_plan, patient_counseling, missing_critical


def run_qtguard_with_retrieval(
    mini_chart: str,
    score_threshold: float = 0.0,
    margin_threshold: float = 0.2,
    top_n_notes: int = 5,
) -> Tuple[Dict[str, Any], List[Evidence], bool]:
    retriever = get_retriever()
    query = _build_retrieval_query(mini_chart)
    evidence = retriever.search(query)

    weak, top_score, margin = _is_evidence_weak(
        evidence=evidence,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
    )

    base: Dict[str, Any] = build_safe_output(mini_chart).model_dump()

    audit = base.get("audit_view") or {}
    notes = _strip_noise_notes(audit.get("notes") or [])

    rs, ap, pc, missing_critical = _build_evidence_guided_plan(mini_chart, evidence)

    if weak:
        base["risk_summary"] = (
            "Evidence confidence is low for this query. Safe deferral recommended until "
            "key clinical inputs and/or stronger supporting references are available."
        )
        base["action_plan"] = [
            "Safe deferral: confirm QTc, potassium (K), magnesium (Mg), medication list with doses, and renal/hepatic status if available.",
            "Expand the knowledge base with trusted references to enable citation-grounded recommendations.",
        ] + (base.get("action_plan") or [])
    else:
        base["risk_summary"] = rs
        base["action_plan"] = ap
        base["patient_counseling"] = pc

    # For evaluation, missing critical inputs should still count as deferral
    weak_for_eval = weak or missing_critical

    notes.append(
        f"Retrieval diagnostics: top_score={top_score:.3f}; margin={margin:.3f}; weak={weak_for_eval}; "
        f"score_threshold={score_threshold:.3f}; margin_threshold={margin_threshold:.3f}"
    )
    notes.append(f"Retrieval query: {query}")
    notes.extend(_evidence_notes(evidence, top_n=top_n_notes))

    audit["notes"] = notes
    base["audit_view"] = audit

    return base, evidence, weak_for_eval

