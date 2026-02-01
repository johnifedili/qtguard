# qtguard_core/eval_harness.py
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from qtguard_core.rag_pipeline import run_qtguard_with_retrieval

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def keyword_hits(hay: str, keywords: List[str]) -> Tuple[int, int, List[str]]:
    h = norm(hay)
    hits = [k for k in keywords if norm(k) in h]
    return len(hits), len(keywords), hits

def is_deferral(output: Dict[str, Any], weak_flag: bool) -> bool:
    rs = norm(output.get("risk_summary", ""))
    ap = " ".join(output.get("action_plan", []) or [])
    apn = norm(ap)
    return weak_flag or ("safe deferral" in rs) or ("safe deferral" in apn) or ("missing key inputs" in apn)

def load_cases(eval_path: Path) -> List[Dict[str, Any]]:
    cases = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases

def run_eval(cases: List[Dict[str, Any]], *, score_threshold=0.0, margin_threshold=0.5, top_n_notes=5):
    # returns (summary_dict, per_case_records)
    # implement using your existing logic
    raise NotImplementedError
