import sys
from pathlib import Path
from datetime import datetime
import json

# Ensure repo root is on sys.path when running `python scripts/eval.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qtguard_core.eval_harness import load_cases, run_eval

EVAL_PATH = Path("assets/eval_cases.jsonl")


def main():
    if not EVAL_PATH.exists():
        raise SystemExit(f"Missing {EVAL_PATH}. Create it first.")

    cases = load_cases(EVAL_PATH)

    print(f"Running eval on {len(cases)} cases...\n")

    summary, per_case = run_eval(
        cases,
        score_threshold=0.0,
        margin_threshold=0.5,
        top_n_notes=5,
    )

    # Print per-case quick view (similar to what you had)
    for r in per_case:
        print(f"[{r['case_id']}]")
        print(f"  expect_deferral={r['expect_deferral']}  got_deferral={r['got_deferral']}  weak_flag={r['weak_flag']}")
        print(f"  evidence_top_score={r['evidence_top_score']}  n_evidence={r['n_evidence']}")
        print(f"  evidence_keyword_recall={r['evidence_keyword_recall']:.2f}  hits={r['evidence_hits']}")
        print(f"  plan_keyword_recall={r['plan_keyword_recall']:.2f}  hits={r['plan_hits']}")
        print("")

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # Write artifacts
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports") / "eval_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    per_case_path = out_dir / "per_case.jsonl"
    with per_case_path.open("w", encoding="utf-8") as f:
        for r in per_case:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved per-case results to: {per_case_path}")
    print(f"Saved summary to:          {summary_path}")


if __name__ == "__main__":
    main()
