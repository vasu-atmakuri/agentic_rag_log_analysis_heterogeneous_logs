"""Aggregates raw run records into the paper's tables and statistical tests.

Outputs (in <results_dir>/analysis/):
  accuracy_by_experiment_dataset_method.csv   accuracy, 95% Wilson and bootstrap intervals
  accuracy_by_category_main.csv               per-category accuracy for main comparisons
  paired_mcnemar_between_methods.csv          exact McNemar tests, Holm-corrected per experiment
  paired_mcnemar_ablation_vs_main.csv         each ablation vs the main run on identical questions
  efficiency_and_grounding.csv                latency, tokens, tool calls, unsupported claims, evidence recall
  second_rater_agreement.json                 Cohen's kappa, if a completed second-rater sheet exists
  latex_tables.tex                            booktabs tables ready for the paper
  analysis_summary.md                         human-readable summary

Usage: python -m experiment_evaluation.statistical_analysis_and_tables --config configs/experiment_matrix_journal_extended.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

BOOTSTRAP_RESAMPLES = 2000
METHOD_LABELS = {
    "code_writing_agent": "Code-writing agent",
    "robust_tool_using_agent": "Robust-tool agent",
    "bm25_line_retrieval": "BM25 lines (top-5)",
    "long_context_uniform": "Long context (uniform)",
    "fixed_size_chunk_rag": "Fixed-size chunk RAG",
    "multi_type_chunk_rag": "Multi-type chunk RAG",
    "tool_using_agent": "Tool-using agent",
    "bm25_line_retrieval_top20": "BM25 lines (top-20)",
    "bm25_line_retrieval_top50": "BM25 lines (top-50)",
    "single_query_generation": "Single-shot query generation",
}
METHOD_ORDER = ["bm25_line_retrieval", "bm25_line_retrieval_top20", "bm25_line_retrieval_top50", "long_context_uniform",
                "fixed_size_chunk_rag", "multi_type_chunk_rag", "single_query_generation", "tool_using_agent",
                "robust_tool_using_agent", "code_writing_agent"]


def load_runs(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "raw_runs", "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec.pop("trace", None)
                rows.append(rec)
    if not rows:
        raise SystemExit("No run records found.")
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    # keep the latest record per run key (a rerun after an error supersedes it)
    ok = ok.sort_values("started_utc").drop_duplicates("run_key", keep="last")
    ok["correct"] = ok["correct"].astype(bool)
    errors = df[(df["status"] == "error") & ~df["run_key"].isin(ok["run_key"])].drop_duplicates("run_key", keep="last").copy()
    if not errors.empty:  # pre-registered rule: unresolved failures count as incorrect answers
        errors["correct"] = False
        errors["has_unsupported_claim"] = False
        ok = pd.concat([ok, errors], ignore_index=True)
    return ok, df


def wilson(k: int, n: int):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson") if n else (np.nan, np.nan)
    return lo, hi


def bootstrap_ci(values: np.ndarray, seed: int = 42):
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def accuracy_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        n, k = len(g), int(g["correct"].sum())
        lo, hi = wilson(k, n)
        blo, bhi = bootstrap_ci(g["correct"].astype(float).to_numpy())
        rows.append(dict(zip(keys, key), n=n, correct=k, accuracy=k / n if n else np.nan, wilson_low=lo, wilson_high=hi,
                         bootstrap_low=blo, bootstrap_high=bhi,
                         unsupported_claim_rate=g["has_unsupported_claim"].mean()))
    return pd.DataFrame(rows)


def paired_mcnemar(a: pd.Series, b: pd.Series) -> dict:
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    both = int((joined.a & joined.b).sum())
    only_a = int((joined.a & ~joined.b).sum())
    only_b = int((~joined.a & joined.b).sum())
    neither = int((~joined.a & ~joined.b).sum())
    p = mcnemar([[both, only_a], [only_b, neither]], exact=True).pvalue if (only_a + only_b) else 1.0
    return {"n_paired": len(joined), "acc_a": joined.a.mean() if len(joined) else np.nan,
            "acc_b": joined.b.mean() if len(joined) else np.nan, "only_a_correct": only_a, "only_b_correct": only_b,
            "p_value_exact": p}


def method_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp, g in df.groupby("experiment"):
        methods = [m for m in METHOD_ORDER if m in set(g["method"])]
        pivot = g.assign(qkey=g["dataset"] + "/" + g["question_id"]).pivot_table(
            index="qkey", columns="method", values="correct", aggfunc="first")
        exp_rows = []
        for i, ma in enumerate(methods):
            for mb in methods[i + 1:]:
                res = paired_mcnemar(pivot[ma].astype("boolean"), pivot[mb].astype("boolean"))
                exp_rows.append({"experiment": exp, "method_a": ma, "method_b": mb, **res})
        if exp_rows:
            p_adj = multipletests([r["p_value_exact"] for r in exp_rows], method="holm")[1]
            for r, p in zip(exp_rows, p_adj):
                r["p_value_holm"] = p
        rows.extend(exp_rows)
    return pd.DataFrame(rows)


def ablation_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mains = df[df["experiment"].str.startswith("main_comparison")]
    others = df[~df["experiment"].str.startswith("main_comparison") & ~df["experiment"].str.startswith("repeat_")]
    for exp, g in others.groupby("experiment"):
        model = g["model"].iloc[0]
        base = mains[mains["model"] == model]
        for method, gm in g.groupby("method"):
            bm = base[base["method"] == method]
            a = gm.set_index(gm["dataset"] + "/" + gm["question_id"])["correct"].astype("boolean")
            b = bm.set_index(bm["dataset"] + "/" + bm["question_id"])["correct"].astype("boolean")
            b = b[b.index.isin(a.index)]
            rows.append({"ablation": exp, "method": method, "model": model, **paired_mcnemar(a, b)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.rename(columns={"acc_a": "acc_ablation", "acc_b": "acc_main", "only_a_correct": "only_ablation_correct",
                                  "only_b_correct": "only_main_correct"})
        out["p_value_holm"] = multipletests(out["p_value_exact"], method="holm")[1]
    return out


def efficiency(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["experiment", "method"]).agg(
        runs=("run_key", "count"), median_wall_seconds=("wall_seconds", "median"),
        p90_wall_seconds=("wall_seconds", lambda s: float(np.percentile(s, 90))),
        mean_prompt_tokens=("prompt_tokens", "mean"), mean_completion_tokens=("completion_tokens", "mean"),
        mean_llm_calls=("llm_calls", "mean"), mean_tool_calls=("tool_calls", "mean"),
        unsupported_claim_rate=("has_unsupported_claim", "mean"),
        mean_unsupported_claims=("unsupported_claims", "mean"),
        mean_key_evidence_recall=("key_evidence_recall", "mean"),
        context_overflow_risk_runs=("context_overflow_risk", "sum"),
    ).reset_index()
    if "hit_iteration_limit" in df:
        limits = df.groupby(["experiment", "method"])["hit_iteration_limit"].apply(
            lambda s: int(s.fillna(False).astype(bool).sum())).reset_index()
        agg = agg.merge(limits, on=["experiment", "method"], how="left")
    return agg


def second_rater_agreement(results_dir: str, df: pd.DataFrame):
    path = os.path.join(results_dir, "second_rater", "second_rater_sheet_completed.csv")
    if not os.path.exists(path):
        return None
    from sklearn.metrics import cohen_kappa_score

    sheet = pd.read_csv(path)
    sheet = sheet.dropna(subset=["second_rater_correct"])
    key = pd.read_csv(os.path.join(results_dir, "second_rater", "second_rater_answer_key.csv"))
    merged = sheet.merge(key, on="item_id")
    rater = merged["second_rater_correct"].astype(int)
    auto = merged["automatic_correct"].astype(int)
    return {"items": int(len(merged)), "agreement": float((rater == auto).mean()),
            "cohen_kappa": float(cohen_kappa_score(auto, rater))}


def latex_main_table(acc: pd.DataFrame, experiment: str, caption: str, label: str) -> str:
    sub = acc[acc["experiment"] == experiment]
    if sub.empty:
        return ""
    datasets = list(dict.fromkeys(sub["dataset"]))
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{@{}l" + "c" * (len(datasets) + 1) + "@{}}", r"\toprule",
             "Method & " + " & ".join(d.replace("_", r"\_") for d in datasets) + r" & Pooled \\", r"\midrule"]
    pooled = accuracy_table(sub_runs_cache[experiment], ["method"]).set_index("method") if experiment in sub_runs_cache else None
    for method in METHOD_ORDER:
        row = sub[sub["method"] == method]
        if row.empty:
            continue
        cells = []
        for d in datasets:
            r = row[row["dataset"] == d]
            cells.append(f'{r["accuracy"].iloc[0]:.2f} ({int(r["correct"].iloc[0])}/{int(r["n"].iloc[0])})' if not r.empty else "--")
        if pooled is not None and method in pooled.index:
            p = pooled.loc[method]
            cells.append(f'{p["accuracy"]:.2f} [{p["wilson_low"]:.2f}, {p["wilson_high"]:.2f}]')
        lines.append(METHOD_LABELS[method] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


sub_runs_cache: dict = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    results_dir = args.results_dir or cfg["results_dir"]
    out_dir = os.path.join(results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    ok, all_records = load_runs(results_dir)
    for exp, g in ok.groupby("experiment"):
        sub_runs_cache[exp] = g

    acc = accuracy_table(ok, ["experiment", "dataset", "method"])
    acc.to_csv(os.path.join(out_dir, "accuracy_by_experiment_dataset_method.csv"), index=False)
    pooled = accuracy_table(ok, ["experiment", "method"])
    pooled.to_csv(os.path.join(out_dir, "accuracy_pooled_by_experiment_method.csv"), index=False)
    mains = ok[ok["experiment"].str.startswith("main_comparison")]
    cat = accuracy_table(mains, ["experiment", "category", "method"])
    cat.to_csv(os.path.join(out_dir, "accuracy_by_category_main.csv"), index=False)
    pairs = method_pairs(mains)
    pairs.to_csv(os.path.join(out_dir, "paired_mcnemar_between_methods.csv"), index=False)
    abl = ablation_pairs(ok)
    abl.to_csv(os.path.join(out_dir, "paired_mcnemar_ablation_vs_main.csv"), index=False)
    eff = efficiency(ok[ok["status"] == "ok"])
    eff.to_csv(os.path.join(out_dir, "efficiency_and_grounding.csv"), index=False)
    kappa = second_rater_agreement(results_dir, ok)
    if kappa:
        with open(os.path.join(out_dir, "second_rater_agreement.json"), "w") as handle:
            json.dump(kappa, handle, indent=2)

    tex = []
    for exp in sorted(mains["experiment"].unique()):
        model = mains[mains["experiment"] == exp]["model"].iloc[0]
        tex.append(latex_main_table(acc, exp, f"Answer accuracy by method and log source ({model}). "
                                    "Pooled column: accuracy with 95\\% Wilson interval.", f"tab:{exp}"))
    with open(os.path.join(out_dir, "latex_tables.tex"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(tex))

    errors = all_records[all_records["status"] == "error"]
    unresolved = errors[~errors["run_key"].isin(ok["run_key"])]
    pd.set_option("display.width", 200)
    summary = ["# Analysis summary", "",
               f"Completed runs: {len(ok)}; unresolved errors: {len(unresolved)}", "",
               "## Pooled accuracy by experiment and method", "",
               pooled[["experiment", "method", "n", "accuracy", "wilson_low", "wilson_high", "unsupported_claim_rate"]]
               .round(3).to_markdown(index=False), "",
               "## Paired exact McNemar tests between methods (Holm-corrected)", "",
               pairs.round(4).to_markdown(index=False) if not pairs.empty else "(none)", "",
               "## Ablations versus main run (paired, Holm-corrected)", "",
               abl.round(4).to_markdown(index=False) if not abl.empty else "(none)", "",
               "## Efficiency and grounding", "", eff.round(3).to_markdown(index=False), ""]
    if kappa:
        summary += ["## Second rater", "", json.dumps(kappa), ""]
    with open(os.path.join(out_dir, "analysis_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary))
    print("\n".join(summary[:8]))
    print(f"\nAll tables written to {out_dir}")


if __name__ == "__main__":
    main()
