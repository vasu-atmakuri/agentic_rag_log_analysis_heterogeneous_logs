"""Additional analyses for the extended study: GEE logistic regression, run-to-run repeatability and paraphrase
consistency. Writes CSV and Markdown files to <results_dir>/analysis/.

  gee_logistic_main_comparison.csv   odds ratios for each method versus fixed-size RAG, adjusted for model and
                                     source, exchangeable correlation clustered by question
  repeatability_across_runs.csv      per model and method: accuracy of each run, agreement of all runs
  paraphrase_consistency.csv         per model and method: accuracy for original and paraphrased wording,
                                     share of questions graded identically under all wordings

Usage: python -m experiment_evaluation.journal_extended_analysis --config configs/experiment_matrix_journal_extended.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml


def load(results_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "raw_runs", "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ok":
                r.pop("trace", None)
                rows.append(r)
    df = pd.DataFrame(rows)
    df["correct"] = df["correct"].astype(int)
    df["qkey"] = df["dataset"] + "/" + df["question_id"]
    return df


def gee_main(df: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    main = df[df["experiment"].str.startswith("main_comparison")].copy()
    if main.empty:
        return pd.DataFrame()
    model = smf.gee("correct ~ C(method, Treatment('fixed_size_chunk_rag')) + C(model) + C(dataset)", "qkey", main,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable())
    fit = model.fit()
    table = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p_value": fit.pvalues})
    ci = fit.conf_int()
    table["odds_ratio"] = np.exp(table["coef"])
    table["or_low"] = np.exp(ci[0])
    table["or_high"] = np.exp(ci[1])
    table = table[table.index.str.contains("method")]
    table.index = table.index.str.replace(r"C\(method, Treatment\('fixed_size_chunk_rag'\)\)\[T\.", "", regex=True).str.rstrip("]")
    return table.reset_index().rename(columns={"index": "method_vs_fixed_size_rag"})


def repeatability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp in sorted(e for e in df["experiment"].unique() if e.startswith("repeat_")):
        rep = df[df["experiment"] == exp]
        model = rep["model"].iloc[0]
        base = df[(df["experiment"].str.startswith("main_comparison")) & (df["model"] == model)]
        for method, g in rep.groupby("method"):
            runs = {0: base[base["method"] == method].set_index("qkey")["correct"]}
            for idx, gg in g.groupby("repeat_index"):
                runs[int(idx) + 1] = gg.set_index("qkey")["correct"]
            frame = pd.DataFrame(runs).dropna()
            if frame.empty:
                continue
            identical = (frame.nunique(axis=1) == 1).mean()
            accs = frame.mean(axis=0)
            rows.append({"model": model, "method": method, "runs": frame.shape[1], "questions": len(frame),
                         "accuracy_per_run": ";".join(f"{a:.3f}" for a in accs), "accuracy_sd": round(accs.std(ddof=1), 4),
                         "share_identical_across_runs": round(identical, 4)})
    return pd.DataFrame(rows)


def paraphrase_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    para = df[df["experiment"].str.startswith("paraphrase_")]
    for (model, method), g in para.groupby(["model", "method"]):
        base = df[(df["experiment"].str.startswith("main_comparison")) & (df["model"] == model) & (df["method"] == method)]
        cols = {"original": base.set_index("qkey")["correct"]}
        for variant, gv in g.groupby("question_variant"):
            cols[variant] = gv.set_index("qkey")["correct"]
        frame = pd.DataFrame(cols).dropna()
        if frame.empty or frame.shape[1] < 2:
            continue
        row = {"model": model, "method": method, "questions": len(frame),
               "share_identical_across_wordings": round((frame.nunique(axis=1) == 1).mean(), 4)}
        row.update({f"accuracy_{c}": round(frame[c].mean(), 4) for c in frame.columns})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = os.path.join(cfg["results_dir"], "analysis")
    os.makedirs(out, exist_ok=True)
    df = load(cfg["results_dir"])
    parts = []
    for name, fn in (("gee_logistic_main_comparison", gee_main), ("repeatability_across_runs", repeatability),
                     ("paraphrase_consistency", paraphrase_consistency)):
        try:
            table = fn(df)
        except Exception as exc:  # noqa: BLE001
            table = pd.DataFrame({"error": [f"{type(exc).__name__}: {exc}"]})
        table.to_csv(os.path.join(out, f"{name}.csv"), index=False)
        parts.append(f"## {name}\n\n" + (table.round(4).to_markdown(index=False) if not table.empty else "(no data)") + "\n")
        print(f"\n== {name}\n{table.round(4).to_string(index=False) if not table.empty else '(no data)'}")
    with open(os.path.join(out, "journal_extended_analysis_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("# Extended analyses\n\n" + "\n".join(parts))


if __name__ == "__main__":
    main()
