cat > qtguard_core/rag_pipeline.py <<'EOF'
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from qtguard_core.guardrails import build_safe_output
from qtguard_core.retrieval import get_retriever, Evidence


def _build_retrieval_query(mini_chart: str) -> str:
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


def run_qtguard_with_retrieval(
    mini_chart: str,
    score_threshold: float = 1.5,
) -> Tuple[Dict[str, Any], List[Evidence], bool]:
    """
    Returns:
      - output dict matching QTGuardOutput-like schema
      - evidence list
      - evidence_is_weak flag

    This is Kaggle-safe: it does not require gated LLM access.
    Output is now explicitly driven by retrieved evidence + safety gating.
    """
    retriever = get_retriever()
    query = _build_retrieval_query(mini_chart)
    evidence = retriever.search(query)

    top_score = evidence[0].score if evidence else float("-inf")
    weak = (not evidence) or (top_score < score_threshold)

    # Start from your existing safe output
    base = build_safe_output(mini_chart).model_dump()

    # Make the OUTPUT explicitly retriever-driven:
    # 1) if weak evidence, force safe deferral language + ask for inputs
    # 2) if strong evidence, keep guardrails but reference evidence + add evidence-aware notes
    if weak:
        base["risk_summary"] = (
            "Evidence confidence is low for this query. Safe deferral recommended until "
            "key clinical inputs and/or better supporting references are available."
        )
        # Keep existing action plan but prepend evidence message
        base["action_plan"] = [
            "Safe deferral: provide QTc, potassium (K), magnesium (Mg), full medication list with doses, and renal/hepatic status if available.",
            "If you can share the relevant guideline or policy source, QTGuard can cite it explicitly.",
        ] + (base.get("action_plan") or [])
    else:
        # Strengthen output with evidence cues (still safe; avoids hallucination)
        base["risk_summary"] = (
            base.get("risk_summary", "")
            + " Evidence was retrieved and reranked to support this triage."
        )
        # Optionally add an evidence-informed first action
        base["action_plan"] = [
            f"Evidence available: see [E1]–[E5] in Audit view (top score={top_score:.3f})."
        ] + (base.get("action_plan") or [])

    # Attach evidence trail to audit view
    audit = base.get("audit_view") or {}
    notes = audit.get("notes") or []
    notes.append(f"Top evidence score={top_score:.3f}; weak={weak}")
    notes.extend(_evidence_notes(evidence))
    audit["notes"] = notes
    base["audit_view"] = audit

    return base, evidence, weak
EOF
