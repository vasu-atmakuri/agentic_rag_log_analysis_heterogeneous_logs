"""Agent failure taxonomy, characterization-version comparison, and paper figures.

Failure categories for incorrect agent answers (first matching rule wins):
  no_tool_call          answered without calling any tool
  iteration_limit       used all agent turns without a final answer
  empty_tool_result     the last tool observation matched no events (wrong time, level or component)
  evidence_not_reached  the question's key evidence lines were never returned by a tool
  evidence_misread      the key evidence lines were returned but the final answer was still wrong
  other_wrong_answer    none of the above (no key evidence defined for the question)

Outputs go to <results_dir>/analysis/.

Usage: python -m experiment_evaluation.failure_taxonomy_and_figures --config configs/experiment_matrix_m5_pro_48gb_characterization_v2.yaml \
           --version1-results experiment_results_m5_pro_48gb
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

METHOD_ORDER = ["bm25_line_retrieval", "long_context_uniform", "fixed_size_chunk_rag", "multi_type_chunk_rag",
                "tool_using_agent"]
METHOD_LABELS = {"bm25_line_retrieval": "BM25 lines", "long_context_uniform": "Long context",
                 "fixed_size_chunk_rag": "Fixed-size RAG", "multi_type_chunk_rag": "Multi-type RAG",
                 "tool_using_agent": "Tool-using agent"}
MODEL_ORDER = [("main_comparison__llama3_2_3b", "Llama 3.2 3B"), ("main_comparison__qwen3_4b_instruct", "Qwen3 4B"),
               ("main_comparison__llama3_1_8b", "Llama 3.1 8B"), ("main_comparison__mistral_small_3_2_24b", "Mistral Small 24B"),
               ("main_comparison__qwen3_30b_a3b_instruct", "Qwen3 30B-A3B")]
CATEGORY_ORDER = ["factual", "temporal", "comparative", "causal"]
FAILURE_ORDER = ["no_tool_call", "iteration_limit", "empty_tool_result", "evidence_not_reached", "evidence_misread",
                 "other_wrong_answer"]


def load_runs(results_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "raw_runs", "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok":
                rows.append(r)
    return rows


def failure_category(r: dict) -> str:
    if r.get("tool_calls", 0) == 0:
        return "no_tool_call"
    if r.get("hit_iteration_limit"):
        return "iteration_limit"
    tools = [t for t in r.get("trace", []) if t.get("type") == "tool"]
    if tools:
        head = tools[-1].get("observation_head", "")
        if "total_matching_events=0" in head or head.startswith("No matching") or head.startswith("No chunks"):
            return "empty_tool_result"
    recall = r.get("key_evidence_recall")
    if recall is not None:
        return "evidence_misread" if recall >= 1.0 else "evidence_not_reached"
    return "other_wrong_answer"


def failure_taxonomy(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for exp, label in MODEL_ORDER:
        agent = [r for r in runs if r["experiment"] == exp and r["method"] == "tool_using_agent"]
        wrong = [r for r in agent if not r["correct"]]
        counts = Counter(failure_category(r) for r in wrong)
        row = {"model": label, "agent_runs": len(agent), "incorrect": len(wrong),
               "zero_tool_call_runs_all": sum(1 for r in agent if r.get("tool_calls", 0) == 0)}
        row.update({c: counts.get(c, 0) for c in FAILURE_ORDER})
        rows.append(row)
    return pd.DataFrame(rows)


def characterization_versions(v1_runs: list[dict], v2_runs: list[dict]) -> pd.DataFrame:
    """Per model and method: accuracy with characterization v1 (silent fallback where it failed) and v2."""
    rows = []
    for exp, label in MODEL_ORDER:
        for method in ("multi_type_chunk_rag", "tool_using_agent"):
            a = {(r["dataset"], r["question_id"]): r["correct"] for r in v1_runs if r["experiment"] == exp and r["method"] == method}
            b = {(r["dataset"], r["question_id"]): r["correct"] for r in v2_runs if r["experiment"] == exp and r["method"] == method}
            common = sorted(set(a) & set(b))
            if not common:
                continue
            rows.append({"model": label, "method": method, "paired_questions": len(common),
                         "accuracy_v1": round(np.mean([a[k] for k in common]), 4),
                         "accuracy_v2": round(np.mean([b[k] for k in common]), 4),
                         "only_v1_correct": sum(a[k] and not b[k] for k in common),
                         "only_v2_correct": sum(b[k] and not a[k] for k in common)})
    g = [r for r in v2_runs if r["experiment"].startswith("characterization_ablation_generic")]
    for method in ("multi_type_chunk_rag", "tool_using_agent"):
        gen = {(r["dataset"], r["question_id"]): r["correct"] for r in g if r["method"] == method}
        auto = {(r["dataset"], r["question_id"]): r["correct"] for r in v2_runs
                if r["experiment"] == "main_comparison__qwen3_4b_instruct" and r["method"] == method}
        common = sorted(set(gen) & set(auto))
        if common:
            rows.append({"model": "Qwen3 4B (generic vs automatic v2)", "method": method, "paired_questions": len(common),
                         "accuracy_v1": round(np.mean([gen[k] for k in common]), 4),
                         "accuracy_v2": round(np.mean([auto[k] for k in common]), 4),
                         "only_v1_correct": sum(gen[k] and not auto[k] for k in common),
                         "only_v2_correct": sum(auto[k] and not gen[k] for k in common)})
    return pd.DataFrame(rows)


def save(fig, out_dir: str, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_accuracy_by_model(runs, out_dir):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.16
    x = np.arange(len(MODEL_ORDER))
    for i, method in enumerate(METHOD_ORDER):
        acc, lo, hi = [], [], []
        for exp, _ in MODEL_ORDER:
            sub = [r["correct"] for r in runs if r["experiment"] == exp and r["method"] == method]
            k, n = sum(sub), len(sub)
            a = k / n if n else 0
            l, h = proportion_confint(k, n, method="wilson") if n else (0, 0)
            acc.append(a), lo.append(a - l), hi.append(h - a)
        ax.bar(x + (i - 2) * width, acc, width, yerr=[lo, hi], capsize=2, label=METHOD_LABELS[method],
               color=plt.cm.Greys(0.25 + 0.15 * i) if method != "tool_using_agent" else "#1f4e79")
    ax.set_xticks(x, [label.replace(" ", "\n", 1) if len(label) > 12 else label for _, label in MODEL_ORDER], fontsize=8)
    ax.set_ylabel("Answer accuracy")
    ax.set_ylim(0, 0.8)
    ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, out_dir, "figure_accuracy_by_method_and_model")


def figure_accuracy_by_category(runs, out_dir):
    exp = "main_comparison__qwen3_4b_instruct"
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    width = 0.16
    x = np.arange(len(CATEGORY_ORDER))
    for i, method in enumerate(METHOD_ORDER):
        vals = []
        for c in CATEGORY_ORDER:
            sub = [r["correct"] for r in runs if r["experiment"] == exp and r["method"] == method and r["category"] == c]
            vals.append(sum(sub) / len(sub) if sub else 0)
        ax.bar(x + (i - 2) * width, vals, width, label=METHOD_LABELS[method],
               color=plt.cm.Greys(0.25 + 0.15 * i) if method != "tool_using_agent" else "#1f4e79")
    ax.set_xticks(x, [c.capitalize() for c in CATEGORY_ORDER])
    ax.set_ylabel("Accuracy (Qwen3 4B)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, out_dir, "figure_accuracy_by_question_category_qwen3_4b")


def figure_cost_accuracy(runs, out_dir):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    markers = dict(zip(METHOD_ORDER, ["o", "s", "^", "D", "*"]))
    for method in METHOD_ORDER:
        for exp, label in MODEL_ORDER:
            sub = [r for r in runs if r["experiment"] == exp and r["method"] == method]
            if not sub:
                continue
            ax.scatter(np.median([r["wall_seconds"] for r in sub]), np.mean([r["correct"] for r in sub]),
                       marker=markers[method], s=60 if method == "tool_using_agent" else 30,
                       color="#1f4e79" if method == "tool_using_agent" else "0.5",
                       label=METHOD_LABELS[method] if exp == MODEL_ORDER[0][0] else None)
    ax.set_xscale("log")
    ax.set_xlabel("Median seconds per question (M5 Pro)")
    ax.set_ylabel("Answer accuracy")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.3)
    save(fig, out_dir, "figure_latency_versus_accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--version1-results", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = os.path.join(cfg["results_dir"], "analysis")
    os.makedirs(out_dir, exist_ok=True)
    v2 = load_runs(cfg["results_dir"])
    v1 = load_runs(args.version1_results)
    tax = failure_taxonomy(v2)
    tax.to_csv(os.path.join(out_dir, "agent_failure_taxonomy.csv"), index=False)
    cv = characterization_versions(v1, v2)
    cv.to_csv(os.path.join(out_dir, "characterization_version_comparison.csv"), index=False)
    figure_accuracy_by_model(v2, out_dir)
    figure_accuracy_by_category(v2, out_dir)
    figure_cost_accuracy(v2, out_dir)
    with open(os.path.join(out_dir, "failure_taxonomy_and_characterization_summary.md"), "w", encoding="utf-8") as h:
        h.write("# Agent failure taxonomy\n\n" + tax.to_markdown(index=False) + "\n\n")
        h.write("# Characterization v1 (silent fallback where it failed) versus v2\n\n" + cv.to_markdown(index=False) + "\n")
    print(tax.to_string(index=False))
    print()
    print(cv.to_string(index=False))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
