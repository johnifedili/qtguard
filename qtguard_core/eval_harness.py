# qtguard_core/eval_harness.py
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from qtguard_core.rag_pipeline import run_qtguard_with_retrieval


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def keyword_match(hay: str, keyword: str) -> bool:
    """
    Keyword matching for eval that supports clinically equivalent phrasing.
    This prevents false "misses" when the plan uses safe synonyms (e.g.,
    "stabilizing electrolytes" vs "Correct electrolytes").
    """
    h = norm(hay)
    k = norm(keyword)

    # Default: substring match
    if k in h:
        return True

    # Synonym-aware matches for specific expected keywords
    if k == "correct electrolytes":
        return re.search(
            r"(correct|replete|replace|optimi[sz]e|stabili[sz]e)\s+electrolytes"
            r"|electrolyte\s+(correction|repletion|optimization)"
            r"|replet(e|ing)\s+(k|potassium)"
            r"|replet(e|ing)\s+(mg|magnesium)"
            r"|correct(ing)?\s+(k|potassium)"
            r"|correct(ing)?\s+(mg|magnesium)",
            h,
        ) is not None

    if k == "missing key inputs":
        return ("missing key inputs" in h) or ("missing required inputs" in h) or ("required inputs" in h and "missing" in h)

    if k == "safe deferral":
        return ("safe deferral" in h) or ("deferral" in h and "safe" in h)

    return False


def keyword_hits(hay: str, keywords: List[str]) -> Tuple[int, int, List[str]]:
    hits = [k for k in keywords if keyword_match(hay, k)]
    return len(hits), len(keywords), hits


def is_deferral(output: Dict[str, Any], weak_flag: bool) -> bool:
    rs = norm(output.get("risk_summary", ""))
    ap = " ".join(output.get("action_plan", []) or [])
    apn = norm(ap)
    return weak_flag or ("safe deferral" in rs) or ("safe deferral" in apn) or ("missing key inputs" in apn)


def load_cases(eval_path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def run_eval(
    cases: List[Dict[str, Any]],
    *,
    score_threshold: float = 0.0,
    margin_threshold: float = 0.5,
    top_n_notes: int = 5,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    n = len(cases)
    deferral_correct = 0
    evidence_keyword_recall_sum = 0.0
    plan_keyword_recall_sum = 0.0
    evidence_hit_cases = 0
    plan_hit_cases = 0

    per_case: List[Dict[str, Any]] = []

    for c in cases:
        case_id = c["case_id"]
        mini_chart = c["mini_chart"]
        expect_def = bool(c.get("expect_deferral", False))
        expected_keywords = c.get("expected_keywords", [])

        out, evidence, weak = run_qtguard_with_retrieval(
            mini_chart,
            score_threshold=score_threshold,
            margin_threshold=margin_threshold,
            top_n_notes=top_n_notes,
        )

        evidence_text = "\n".join([getattr(e, "text", "") for e in (evidence or [])])
        plan_text = "\n".join(out.get("action_plan", []) or [])
        rs_text = out.get("risk_summary", "")

        got_def = is_deferral(out, weak)
        deferral_correct += int(got_def == expect_def)

        e_hits, e_total, e_hit_list = keyword_hits(evidence_text, expected_keywords)
        e_recall = (e_hits / e_total) if e_total else 1.0
        evidence_keyword_recall_sum += e_recall
        evidence_hit_cases += int(e_hits > 0)

        p_hits, p_total, p_hit_list = keyword_hits(plan_text + "\n" + rs_text, expected_keywords)
        p_recall = (p_hits / p_total) if p_total else 1.0
        plan_keyword_recall_sum += p_recall
        plan_hit_cases += int(p_hits > 0)

        per_case.append({
            "case_id": case_id,
            "mini_chart": mini_chart,
            "expect_deferral": expect_def,
            "got_deferral": got_def,
            "weak_flag": weak,
            "expected_keywords": expected_keywords,
            "evidence_keyword_recall": round(e_recall, 4),
            "plan_keyword_recall": round(p_recall, 4),
            "evidence_hits": e_hit_list,
            "plan_hits": p_hit_list,
            "evidence_top_score": (getattr(evidence[0], "score", None) if evidence else None),
            "n_evidence": len(evidence) if evidence else 0,
            "output": out,
            "evidence": [
                {
                    "rank": i + 1,
                    "score": getattr(e, "score", None),
                    "chunk_id": getattr(e, "chunk_id", None) or getattr(e, "id", None) or getattr(e, "source_id", None),
                    "source": getattr(e, "source", None) or getattr(e, "doc", None) or getattr(e, "path", None),
                }
                for i, e in enumerate(evidence or [])
            ],
        })

    deferral_acc = deferral_correct / n if n else 0.0
    avg_evidence_recall = evidence_keyword_recall_sum / n if n else 0.0
    avg_plan_recall = plan_keyword_recall_sum / n if n else 0.0
    composite = 0.5 * deferral_acc + 0.25 * avg_evidence_recall + 0.25 * avg_plan_recall

    summary = {
        "n_cases": n,
        "deferral_accuracy": round(deferral_acc, 4),
        "avg_evidence_keyword_recall": round(avg_evidence_recall, 4),
        "avg_plan_keyword_recall": round(avg_plan_recall, 4),
        "cases_with_evidence_hit": evidence_hit_cases,

        "cases_with_plan_hit": plan_hit_cases,
        "composite": round(composite, 4),
    }

    return summary, per_case




