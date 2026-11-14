"""Summarizes the revision experiments (BM25 with 20 and 50 lines, single-shot query generation, timeline ablation
with Qwen3-30B-A3B) against the main-comparison runs of the same questions and models.

Usage: python -m experiment_evaluation.revision_experiments_summary > experiment_results_revision_experiments/analysis/revision_experiments_summary.txt
"""
import collections
import json
import math

MODELS = ["llama3_2_3b", "qwen3_4b_instruct", "llama3_1_8b", "mistral_small_3_2_24b", "qwen3_30b_a3b_instruct"]
SINGLE_RECORD = {"factual", "event_precedence", "causal_diagnosis", "precedence", "causal"}
SETS = [("template", "experiment_results_revision_experiments/raw_runs/revision_baselines__{}.jsonl",
         "experiment_results_journal_extended/raw_runs/main_comparison__{}.jsonl"),
        ("practitioner", "experiment_results_revision_experiments_practitioner/raw_runs/revision_baselines_practitioner__{}.jsonl",
         "experiment_results_engineer_written/raw_runs/engineer_written__{}.jsonl")]


def load(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def mcnemar(a, b):
    n10 = sum(1 for q in a if a[q] and not b[q])
    n01 = sum(1 for q in a if b[q] and not a[q])
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    return n10, n01, min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n)


def primary(r):
    return (str(r.get("repeat_index", 0)) == "0" and r.get("question_variant") in (None, "None")
            and not r.get("disabled_tools") and r.get("characterization", "generic") == "generic")


def main():
    for name, rev, base in SETS:
        print(f"===== {name}")
        for model in MODELS:
            res, cat = collections.defaultdict(dict), {}
            for r in load(base.format(model)) + load(rev.format(model)):
                if primary(r):
                    res[r["method"]][r["question_id"]] = bool(r.get("correct", False))
                    cat[r["question_id"]] = r.get("category")
            acc = {m: round(100 * sum(v.values()) / len(v), 1) for m, v in res.items()}
            print(model, {m: acc[m] for m in sorted(acc)})
            for m in ["bm25_line_retrieval_top20", "bm25_line_retrieval_top50", "single_query_generation"]:
                print(f"   {m}: vs BM25-5 {mcnemar(res[m], res['bm25_line_retrieval'])}"
                      f" vs robust agent {mcnemar(res[m], res['robust_tool_using_agent'])}"
                      f" vs code agent {mcnemar(res[m], res['code_writing_agent'])}")
            if name == "template":
                for m in ["bm25_line_retrieval", "bm25_line_retrieval_top20", "bm25_line_retrieval_top50",
                          "single_query_generation", "robust_tool_using_agent", "code_writing_agent"]:
                    s = [v for q, v in res[m].items() if cat[q] in SINGLE_RECORD]
                    mu = [v for q, v in res[m].items() if cat[q] not in SINGLE_RECORD]
                    print(f"   split {m}: single {100*sum(s)/len(s):.1f} ({len(s)}) multi {100*sum(mu)/len(mu):.1f} ({len(mu)})")
    abl = {r["question_id"]: bool(r.get("correct", False)) for r in
           load("experiment_results_revision_experiments/raw_runs/tool_ablation_without_get_timeline__qwen3_30b_a3b_instruct.jsonl")}
    full = {r["question_id"]: bool(r.get("correct", False)) for r in
            load("experiment_results_journal_extended/raw_runs/main_comparison__qwen3_30b_a3b_instruct.jsonl")
            if r["method"] == "tool_using_agent"}
    print(f"timeline ablation Qwen3-30B-A3B: full {100*sum(full.values())/len(full):.1f} without timeline "
          f"{100*sum(abl.values())/len(abl):.1f} discordant/p {mcnemar(full, abl)}")


if __name__ == "__main__":
    main()
