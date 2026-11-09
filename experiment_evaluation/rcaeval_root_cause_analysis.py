"""Analysis of the RCAEval RE3 root-cause localization runs (AC@1 of each model and method).

Writes to <results_dir>/analysis/:
  rcaeval_accuracy_by_model_and_method.csv     AC@1 with Wilson and bootstrap intervals
  rcaeval_accuracy_by_system.csv               AC@1 per microservice system
  rcaeval_accuracy_by_fault.csv                AC@1 per root-cause service and fault id (all models pooled)
  rcaeval_method_comparisons.csv               exact McNemar tests between methods, Holm-corrected per model
  rcaeval_root_cause_summary.md                the tables above plus the non-model baselines

Usage: python -m experiment_evaluation.rcaeval_root_cause_analysis --config configs/experiment_matrix_rcaeval_re3.yaml
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml

from .statistical_analysis_and_tables import METHOD_LABELS, METHOD_ORDER, accuracy_table, load_runs, method_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = os.path.join(cfg["results_dir"], "analysis")
    os.makedirs(out, exist_ok=True)
    df, _ = load_runs(cfg["results_dir"])
    df["system"] = df["dataset"].str.extract(r"^rcaeval_re3_(ob|ss|tt)_")[0]
    df["fault"] = df["dataset"].str.extract(r"^rcaeval_re3_(?:ob|ss|tt)_(.+)_run\d+$")[0]
    order = {m: i for i, m in enumerate(METHOD_ORDER)}

    def ordered(table):
        return table.assign(_o=table["method"].map(order)).sort_values(["model", "_o"]).drop(columns="_o")

    by_model = ordered(accuracy_table(df, ["model", "method"]))
    by_system = ordered(accuracy_table(df, ["model", "method", "system"]))
    by_fault = accuracy_table(df, ["method", "fault"])
    pairs = method_pairs(df)
    for name, table in (("rcaeval_accuracy_by_model_and_method", by_model), ("rcaeval_accuracy_by_system", by_system),
                        ("rcaeval_accuracy_by_fault", by_fault), ("rcaeval_method_comparisons", pairs)):
        table.to_csv(os.path.join(out, f"{name}.csv"), index=False)

    parts = ["# RCAEval RE3 root-cause localization (AC@1)\n"]
    view = by_model.assign(method=by_model["method"].map(METHOD_LABELS))[
        ["model", "method", "n", "correct", "accuracy", "wilson_low", "wilson_high"]]
    parts.append("## By model and method\n\n" + view.round(3).to_markdown(index=False) + "\n")
    pivot = by_system.pivot_table(index=["model", "method"], columns="system", values="accuracy").round(3)
    parts.append("## By system\n\n" + pivot.to_markdown() + "\n")
    baseline_path = os.path.join(out, "rcaeval_heuristic_baselines_summary.csv")
    if os.path.exists(baseline_path):
        parts.append("## Non-model baselines\n\n" + pd.read_csv(baseline_path).to_markdown(index=False) + "\n")
    if not pairs.empty:
        agent_rows = pairs[pairs["method_a"].str.contains("agent") | pairs["method_b"].str.contains("agent")]
        parts.append("## Paired comparisons involving agents\n\n" + agent_rows.round(4).to_markdown(index=False) + "\n")
    with open(os.path.join(out, "rcaeval_root_cause_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
