"""Summarizes the second-round review experiments.

(1) Parser sensitivity: accuracy of the three agents on the HealthApp, Mac and HPC questions with parser revision 1
    (main comparison) and revision 2 (corrected), per model, with exact McNemar tests.
(2) Budget control: three independently sampled single-shot queries per question; accuracy of each sample, of the
    majority answer (ties broken by the first sample) and of "any sample correct", against the iterative code agent.

Usage: python -m experiment_evaluation.second_round_review_summary > experiment_results_parser_revision_2_sensitivity/analysis/second_round_review_summary.txt
"""
import collections
import json
import math
import os
import re

MODELS = ["llama3_2_3b", "qwen3_4b_instruct", "llama3_1_8b", "mistral_small_3_2_24b", "qwen3_30b_a3b_instruct"]
AGENTS = ["tool_using_agent", "robust_tool_using_agent", "code_writing_agent"]
SOURCES = {"loghub2_healthapp", "loghub2_mac", "loghub2_hpc"}


def load(path):
    if not os.path.exists(path):
        return []
    last = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            last[r["run_key"]] = r
    return list(last.values())


def mcnemar(a, b):
    n10 = sum(1 for q in a if q in b and a[q] and not b[q])
    n01 = sum(1 for q in a if q in b and b[q] and not a[q])
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    return n10, n01, min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n)


def primary(r):
    return (str(r.get("repeat_index", 0)) == "0" and r.get("question_variant") in (None, "None")
            and not r.get("disabled_tools") and r.get("characterization", "generic") == "generic")


def norm(answer):
    return re.sub(r"[^a-z0-9:.\- ]", "", str(answer or "").lower()).strip()


def main():
    print("== Parser sensitivity: HealthApp, Mac, HPC (revision 1 = main study, revision 2 = corrected)")
    pooled = collections.defaultdict(lambda: [0, 0, 0])
    for m in MODELS:
        v1 = collections.defaultdict(dict)
        for r in load(f"experiment_results_journal_extended/raw_runs/main_comparison__{m}.jsonl"):
            if primary(r) and r["dataset"] in SOURCES and r["method"] in AGENTS:
                v1[r["method"]][r["question_id"]] = bool(r.get("correct", False))
        v2 = collections.defaultdict(dict)
        for r in load(f"experiment_results_parser_revision_2_sensitivity/raw_runs/parser_revision_2__{m}.jsonl"):
            v2[r["method"]][r["question_id"]] = bool(r.get("correct", False))
        for meth in AGENTS:
            a, b = v1[meth], v2[meth]
            if not b:
                continue
            common = [q for q in a if q in b]
            acc1 = 100 * sum(a[q] for q in common) / len(common)
            acc2 = 100 * sum(b[q] for q in common) / len(common)
            print(f"{m:26s} {meth:26s} n={len(common)} rev1={acc1:5.1f} rev2={acc2:5.1f} diff={acc2-acc1:+5.1f} "
                  f"discordant/p={mcnemar(b, a)}")
            pooled[meth][0] += sum(a[q] for q in common)
            pooled[meth][1] += sum(b[q] for q in common)
            pooled[meth][2] += len(common)
    for meth, (c1, c2, n) in pooled.items():
        print(f"pooled {meth:26s} n={n} rev1={100*c1/n:5.1f} rev2={100*c2/n:5.1f} diff={100*(c2-c1)/n:+5.1f}")

    print("\n== Budget control: three sampled single-shot queries (temperature 0.7, seeds 1-3)")
    for m in ["qwen3_30b_a3b_instruct", "qwen3_4b_instruct"]:
        samples = collections.defaultdict(dict)
        for seed in (1, 2, 3):
            for r in load(f"experiment_results_independent_query_samples/raw_runs/single_query_sample_seed{seed}__{m}.jsonl"):
                samples[r["question_id"]][seed] = r
        complete = {q: s for q, s in samples.items() if len(s) == 3}
        if not complete:
            continue
        per_seed = [100 * sum(bool(s[k].get("correct")) for s in complete.values()) / len(complete) for k in (1, 2, 3)]
        vote, anyc = {}, {}
        for q, s in complete.items():
            answers = [norm(s[k].get("final_answer")) for k in (1, 2, 3)]
            counts = collections.Counter(answers)
            top = max(counts.values())
            chosen = next(k for k, a in zip((1, 2, 3), answers) if counts[a] == top)
            vote[q] = bool(s[chosen].get("correct"))
            anyc[q] = any(bool(s[k].get("correct")) for k in (1, 2, 3))
        it = {}
        for r in load(f"experiment_results_journal_extended/raw_runs/main_comparison__{m}.jsonl"):
            if primary(r) and r["method"] == "code_writing_agent":
                it[r["question_id"]] = bool(r.get("correct", False))
        greedy = {}
        for r in load(f"experiment_results_revision_experiments/raw_runs/revision_baselines__{m}.jsonl"):
            if r["method"] == "single_query_generation":
                greedy[r["question_id"]] = bool(r.get("correct", False))
        n = len(complete)
        code_runs = [float(s[k].get("tool_calls") or 0) for s in complete.values() for k in (1, 2, 3)]
        print(f"{m}: questions={n} per-sample={[round(x,1) for x in per_seed]} majority={100*sum(vote.values())/n:.1f} "
              f"any-correct={100*sum(anyc.values())/n:.1f} greedy-single={100*sum(greedy[q] for q in complete)/n:.1f} "
              f"iterative-code-agent={100*sum(it[q] for q in complete)/n:.1f} "
              f"iterative vs majority {mcnemar(it, vote)} iterative vs any {mcnemar(it, anyc)} "
              f"mean query executions per question (3 samples)={3*sum(code_runs)/len(code_runs):.2f}")


if __name__ == "__main__":
    main()
