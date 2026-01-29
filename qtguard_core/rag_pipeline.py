from __future__ import annotations

from typing import Any, Dict, List, Tuple

from qtguard_core.guardrails import build_safe_output
from qtguard_core.retrieval import get_retriever, Evidence


def _build_retrieval_query(mini_chart: str) -> str:
    """
    Simple first-pass query. Later you can parse meds/QTc explicitly,
    but this works well enough to start.
    """
    return f"QT prolongation torsades risk triage guidance. {mini_chart}"


def _evidence_notes(evidence: List[Evidence], top_n: int = 5) -> List[str]:
    notes: List[str] = []
    if not evidence:
        notes.append("No evidence retrieved.")
        return notes

    notes.append("Evidence retrieved (reranked):")
    for i, e in enumerate(evidence[:top_n], start=1):
        notes.append(f"[E{i}] {e.title} — {e.section} — score={e.score:.3f} — {e.chunk_id}")
    return notes


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
      - top score is not clearly better than second score (low margin)

    Returns: (weak, top_score, margin)
    """
    if not evidence:
        return True, float("-inf"), 0.0

    top = evidence[0].score
    second = evidence[1].score if len(evidence) > 1 else float("-inf")
    margin = top - second if second != float("-inf") else float("inf")

    weak = (top < score_threshold) or (margin < margin_threshold)
    return weak, top, margin


def run_qtguard_with_retrieval(
    mini_chart: str,
    score_threshold: float = -5.0,
    margin_threshold: float = 0.5,
    top_n_notes: int = 5,
) -> Tuple[Dict[str, Any], List[Evidence], bool]:
    """
    Retrieval-driven output (Kaggle-safe):
    - Retrieves evidence (BM25 + vector) and reranks it (cross-encoder)
    - Applies an evidence confidence gate (threshold + margin)
    - Produces a schema-compatible output dict + evidence list + weak flag

    NOTE: This does NOT require gated LLM access.
    """
    retriever = get_retriever()
    query = _build_retrieval_query(mini_chart)
    evidence = retriever.search(query)

    weak, top_score, margin = _is_evidence_weak(
        evidence=evidence,
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
    )

    # Start from your existing safe output structure
    base = build_safe_output(mini_chart).model_dump()

    # ---- CLEAN AUDIT NOTES: remove legacy MedGemma gated-repo noise ----
    audit = base.get("audit_view") or {}
    notes = audit.get("notes") or []
    notes = [
        n for n in notes
        if "MedGemma inference error" not in n
        and "gated repo" not in n
        and "Cannot access gated repo" not in n
        and "401 Client Error" not in n
    ]
    # -------------------------------------------------------------------

    # Make output explicitly retrieval-aware
    if weak:
        base["risk_summary"] = (
            "Evidence confidence is low for this query. Safe deferral recommended until "
            "key clinical inputs and/or better supporting references are available."
        )

        base["action_plan"] = [
            "Safe deferral: confirm QTc, potassium (K), magnesium (Mg), full medication list with doses, and renal/hepatic status if available.",
            "Add/expand trusted references in the knowledge base to enable citation-grounded recommendations.",
        ] + (base.get("action_plan") or [])
    else:
        base["risk_summary"] = (
            (base.get("risk_summary") or "").rstrip()
            + " Evidence was retrieved and reranked to support this triage."
        )
        base["action_plan"] = [
            f"Evidence available: see [E1]–[E{min(top_n_notes, len(evidence))}] in Audit view (top score={top_score:.3f}, margin={margin:.3f})."
        ] + (base.get("action_plan") or [])

    # Attach retrieval diagnostics + evidence trail to audit view
    notes.append(
        f"Top evidence score={top_score:.3f}; margin={margin:.3f}; weak={weak}; "
        f"score_threshold={score_threshold:.3f}; margin_threshold={margin_threshold:.3f}"
    )
    notes.append(f"Retrieval query: {query}")
    notes.extend(_evidence_notes(evidence, top_n=top_n_notes))

    audit["notes"] = notes
    base["audit_view"] = audit

    return base, evidence, weak
