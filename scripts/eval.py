import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running `python scripts/eval.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from qtguard_core.rag_pipeline import run_qtguard_with_retrieval


EVAL_PATH = Path("assets/eval_cases.jsonl")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def contains_any(hay: str, needles: List[str]) -> bool:
    h = norm(hay)
    return any(norm(n) in h for n in needles)


def keyword_hits(hay: str, keywords: List[str]) -> Tuple[int, int, List[str]]:
    h = norm(hay)
    hits = [k for k in keywords if norm(k) in h]
    return len(hits), len(keywords), hits


def is_deferral(output: Dict[str, Any], weak_flag: bool) -> bool:
    rs = norm(output.get("risk_summary", ""))
    ap = " ".join(output.get("action_plan", []) or [])
    apn = norm(ap)
    # Robust detection: either pipeline weak flag OR explicit deferral language
    return weak_flag or ("safe deferral" in rs) or ("safe deferral" in apn) or ("missing key inputs" in apn)


def main():
    if not EVAL_PATH.exists():
        raise SystemExit(f"Missing {EVAL_PATH}. Create it first.")

    cases = []
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    # Metrics accumulators
    n = len(cases)
    deferral_correct = 0
    evidence_keyword_recall_sum = 0.0
    plan_keyword_recall_sum = 0.0
    evidence_hit_cases = 0
    plan_hit_cases = 0

    print(f"Running eval on {n} cases...\n")

    for c in cases:
        case_id = c["case_id"]
        mini_chart = c["mini_chart"]
        expect_def = bool(c.get("expect_deferral", False))
        expected_keywords = c.get("expected_keywords", [])

        out, evidence, weak = run_qtguard_with_retrieval(
            mini_chart,
            score_threshold=0.0,   # keep consistent with your calibrated default
            margin_threshold=0.5,
            top_n_notes=5,
        )

        # Build text fields for checking
        evidence_text = "\n".join([e.text for e in (evidence or [])])
        plan_text = "\n".join(out.get("action_plan", []) or [])
        rs_text = out.get("risk_summary", "")

        # Deferral accuracy
        got_def = is_deferral(out, weak)
        deferral_correct += int(got_def == expect_def)

        # Keyword recall in evidence
        e_hits, e_total, e_hit_list = keyword_hits(evidence_text, expected_keywords)
        e_recall = (e_hits / e_total) if e_total else 1.0
        evidence_keyword_recall_sum += e_recall
        evidence_hit_cases += int(e_hits > 0)

        # Keyword recall in plan + risk summary
        p_hits, p_total, p_hit_list = keyword_hits(plan_text + "\n" + rs_text, expected_keywords)
        p_recall = (p_hits / p_total) if p_total else 1.0
        plan_keyword_recall_sum += p_recall
        plan_hit_cases += int(p_hits > 0)

        print(f"[{case_id}]")
        print(f"  expect_deferral={expect_def}  got_deferral={got_def}  weak_flag={weak}")
        print(f"  evidence_top_score={(evidence[0].score if evidence else None)}  n_evidence={len(evidence) if evidence else 0}")
        print(f"  evidence_keyword_recall={e_recall:.2f}  hits={e_hit_list}")
        print(f"  plan_keyword_recall={p_recall:.2f}  hits={p_hit_list}")
        print("")

    # Summary
    deferral_acc = deferral_correct / n if n else 0.0
    avg_evidence_recall = evidence_keyword_recall_sum / n if n else 0.0
    avg_plan_recall = plan_keyword_recall_sum / n if n else 0.0

    print("=== SUMMARY ===")
    print(f"Deferral accuracy:         {deferral_acc:.2f} ({deferral_correct}/{n})")
    print(f"Evidence keyword recall:   {avg_evidence_recall:.2f} (avg over cases)")
    print(f"Plan+summary keyword recall:{avg_plan_recall:.2f} (avg over cases)")
    print(f"Cases w/ >=1 evidence hit: {evidence_hit_cases}/{n}")
    print(f"Cases w/ >=1 plan hit:     {plan_hit_cases}/{n}")

    # Simple composite score (tweak if desired)
    composite = 0.5 * deferral_acc + 0.25 * avg_evidence_recall + 0.25 * avg_plan_recall
    print(f"Composite score (0–1):     {composite:.2f}")


if __name__ == "__main__":
    main()
