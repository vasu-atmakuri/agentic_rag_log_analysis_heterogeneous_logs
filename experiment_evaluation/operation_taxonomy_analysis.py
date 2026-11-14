"""Accuracy by the operation required to establish the answer (second revision taxonomy), and accuracy on the
556 Loghub-2.0 questions for which the exact-computation ceiling was measured.

Operation classes (assigned by template):
  direct_lookup     find the record that states the fact (start-up settings, identifiers, versions)
  ordered_lookup    filter a class of events and select the earliest (overall, per component or after a time)
  event_precedence  find the first event of a class after a time, then its immediate predecessor
  aggregation       filter and count, or find the hour with the most events
  comparison        aggregate per component and compare
  causal_diagnosis  relate several events of an injected incident (root component, root cause, recovery)

Usage: python -m experiment_evaluation.operation_taxonomy_analysis > experiment_results_revision_experiments/analysis/operation_taxonomy_summary.txt
"""
import collections
import json
import math
import random

OPERATION_OF_TEMPLATE = {
    "configured_batch_size_at_start": "direct_lookup", "infrastructure_identifiers": "direct_lookup",
    "model_load_duration": "direct_lookup", "model_version_at_start": "direct_lookup",
    "model_version_during_incident": "direct_lookup", "scheduler_workers_at_start": "direct_lookup",
    "first_component_of_class": "ordered_lookup", "first_component_of_class_after_time": "ordered_lookup",
    "first_time_of_class": "ordered_lookup", "first_time_of_class_for_component": "ordered_lookup",
    "component_immediately_preceding_first_class_event": "event_precedence",
    "count_class_in_window": "aggregation", "count_component_events_in_window": "aggregation",
    "peak_hour_of_class_on_day": "aggregation",
    "top_component_for_class": "comparison", "top_component_in_window": "comparison",
    "which_of_two_components_more_class": "comparison",
    "incident_root_component": "causal_diagnosis", "incident_root_cause": "causal_diagnosis",
    "incident_resolution": "causal_diagnosis",
}
ORDER = ["direct_lookup", "ordered_lookup", "event_precedence", "aggregation", "comparison", "causal_diagnosis"]
MODELS = ["llama3_2_3b", "qwen3_4b_instruct", "llama3_1_8b", "mistral_small_3_2_24b", "qwen3_30b_a3b_instruct"]
METHODS = ["bm25_line_retrieval", "bm25_line_retrieval_top50", "fixed_size_chunk_rag", "multi_type_chunk_rag",
           "long_context_uniform", "tool_using_agent", "robust_tool_using_agent", "code_writing_agent",
           "single_query_generation"]


def load(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def primary(r):
    return (str(r.get("repeat_index", 0)) == "0" and r.get("question_variant") in (None, "None")
            and not r.get("disabled_tools") and r.get("characterization", "generic") == "generic")


def results():
    res = {}
    for m in MODELS:
        for path in (f"experiment_results_journal_extended/raw_runs/main_comparison__{m}.jsonl",
                     f"experiment_results_revision_experiments/raw_runs/revision_baselines__{m}.jsonl"):
            for r in load(path):
                if primary(r):
                    res[(m, r["method"], r["question_id"])] = (bool(r.get("correct", False)), r["template"], r["dataset"])
    return res


def clustered_ci(values_by_question, reps=10000, seed=7):
    """Bootstrap over questions of the mean accuracy (values pooled over models per question)."""
    qs = list(values_by_question)
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        sample = [values_by_question[rng.choice(qs)] for _ in qs]
        stats.append(sum(sum(v) for v in sample) / sum(len(v) for v in sample))
    stats.sort()
    return stats[int(0.025 * reps)], stats[int(0.975 * reps)]


def main():
    res = results()
    print("Accuracy (%) by operation class, mean over the five models (questions per class in brackets)")
    counts = collections.Counter()
    for (m, meth, q), (_, tpl, _) in res.items():
        if m == MODELS[0] and meth == "bm25_line_retrieval":
            counts[OPERATION_OF_TEMPLATE[tpl]] += 1
    print("method".ljust(28) + "".join(f"{c}({counts[c]})".rjust(22) for c in ORDER))
    for meth in METHODS:
        row = []
        for c in ORDER:
            by_q = collections.defaultdict(list)
            for (m, me, q), (ok, tpl, _) in res.items():
                if me == meth and OPERATION_OF_TEMPLATE[tpl] == c:
                    by_q[q].append(ok)
            acc = sum(sum(v) for v in by_q.values()) / sum(len(v) for v in by_q.values())
            lo, hi = clustered_ci(by_q, reps=2000)
            row.append(f"{100*acc:5.1f} [{100*lo:4.1f},{100*hi:4.1f}]")
        print(meth.ljust(28) + "".join(x.rjust(22) for x in row))
    print("\nPer model, accuracy (%) by operation class")
    for m in MODELS:
        print(m)
        for meth in ["bm25_line_retrieval", "bm25_line_retrieval_top50", "robust_tool_using_agent", "code_writing_agent"]:
            row = []
            for c in ORDER:
                v = [ok for (mm, me, q), (ok, tpl, _) in res.items() if mm == m and me == meth and OPERATION_OF_TEMPLATE[tpl] == c]
                row.append(f"{100*sum(v)/len(v):5.1f}")
            print("  " + meth.ljust(28) + " ".join(row))
    print("\nAccuracy (%) on the 556 Loghub-2.0 questions (same population as the exact-computation ceiling, 553/556 = 99.5%)")
    for m in MODELS:
        row = []
        for meth in ["bm25_line_retrieval", "bm25_line_retrieval_top50", "tool_using_agent", "robust_tool_using_agent", "code_writing_agent"]:
            v = [ok for (mm, me, q), (ok, tpl, ds) in res.items() if mm == m and me == meth and ds.startswith("loghub2_")]
            row.append(f"{meth}={100*sum(v)/len(v):.1f} (n={len(v)})")
        print(m, "; ".join(row))


if __name__ == "__main__":
    main()
