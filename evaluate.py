"""Run deterministic offline evaluation and write measured results to evaluation_results.json."""
import json
from pathlib import Path
from app import offline_answer

ROOT = Path(__file__).parent
cases = json.loads((ROOT/"data"/"evaluation.json").read_text(encoding="utf-8"))
results = []
for case in cases:
    response = offline_answer(case["question"])
    status_ok = response["status"] == case["expected_status"]
    answer_lower = response["answer"].lower()
    # Keyword scoring is a transparent exact substring check; not a semantic judge.
    keyword_hits = sum(1 for k in case["expected_keywords"] if k.lower() in answer_lower)
    keyword_total = len(case["expected_keywords"])
    keyword_score = keyword_hits / keyword_total if keyword_total else 1.0
    results.append({
        "id":case["id"], "question":case["question"],
        "expected_status":case["expected_status"], "actual_status":response["status"],
        "status_pass":status_ok, "keyword_hits":keyword_hits, "keyword_total":keyword_total,
        "keyword_score":round(keyword_score,3), "mode":response["mode"],
        "input_tokens_estimate":response["input_tokens"], "output_tokens_estimate":response["output_tokens"],
        "estimated_cost_usd":response["estimated_cost_usd"]
    })
n = len(results)
summary = {
    "evaluation_type":"deterministic offline baseline; no external model used",
    "cases":n,
    "status_accuracy":round(sum(r["status_pass"] for r in results)/n,4) if n else 0,
    "keyword_recall_macro":round(sum(r["keyword_score"] for r in results)/n,4) if n else 0,
    "total_keyword_hits":sum(r["keyword_hits"] for r in results),
    "total_expected_keywords":sum(r["keyword_total"] for r in results),
    "estimated_input_tokens":sum(r["input_tokens_estimate"] for r in results),
    "estimated_output_tokens":sum(r["output_tokens_estimate"] for r in results),
    "estimated_cost_usd":round(sum(r["estimated_cost_usd"] for r in results),8),
    "limitations":"Keyword overlap and substring scoring are simple baseline metrics, not a semantic correctness or safety certification."
}
out = {"summary":summary,"results":results}
(ROOT/"evaluation_results.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(summary,indent=2))
print(f"\nSaved {n} case results to evaluation_results.json")
