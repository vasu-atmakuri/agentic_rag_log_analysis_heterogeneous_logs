"""Agreement between the two human raters and the automatic grader.

Reads the returned sheets from <extended results_dir>/human_validation/completed/ (human_validation_sheet_rater_a
and _rater_b, .xlsx or .csv) and the private answer key, then reports raw agreement and Cohen's kappa for rater A
vs rater B, each rater vs the automatic grade, and the items both raters agree on vs the automatic grade, overall
and by answer type and source group. Items where both raters disagree with the automatic grade are listed.

Usage: python -m experiment_evaluation.human_validation_agreement --config configs/experiment_matrix_journal_extended.yaml
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score


def read_sheet(folder: str, rater: str) -> pd.DataFrame:
    for ext in (".xlsx", ".csv"):
        path = os.path.join(folder, f"human_validation_sheet_rater_{rater}{ext}")
        if os.path.exists(path):
            frame = pd.read_excel(path, sheet_name="Items") if ext == ".xlsx" else pd.read_csv(path)
            frame = frame[["item_id", "rater_correct", "notes"]].rename(
                columns={"rater_correct": f"rater_{rater}", "notes": f"notes_{rater}"})
            return frame.dropna(subset=[f"rater_{rater}"])
    raise SystemExit(f"Completed sheet for rater {rater} not found in {folder}")


def pair(a: pd.Series, b: pd.Series) -> dict:
    return {"items": int(len(a)), "agreement": round(float((a == b).mean()), 4),
            "cohen_kappa": round(float(cohen_kappa_score(a, b)), 4) if len(set(a) | set(b)) > 1 else float("nan")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    folder = os.path.join(cfg["results_dir"], "human_validation")
    key = pd.read_csv(os.path.join(folder, "human_validation_answer_key.csv"))
    merged = key.merge(read_sheet(os.path.join(folder, "completed"), "a"), on="item_id").merge(
        read_sheet(os.path.join(folder, "completed"), "b"), on="item_id")
    for c in ("rater_a", "rater_b", "automatic_correct"):
        merged[c] = merged[c].astype(int)
    merged["source_group"] = merged["experiment"].str.startswith("rcaeval").map({True: "rcaeval_re3", False: "extended_sources"})

    rows = []
    for label, g in [("all", merged)] + [(f"answer_type={k}", v) for k, v in merged.groupby("answer_type")] + \
                    [(f"source={k}", v) for k, v in merged.groupby("source_group")]:
        consensus = g[g["rater_a"] == g["rater_b"]]
        rows += [{"subset": label, "comparison": "rater A vs rater B", **pair(g["rater_a"], g["rater_b"])},
                 {"subset": label, "comparison": "rater A vs automatic", **pair(g["rater_a"], g["automatic_correct"])},
                 {"subset": label, "comparison": "rater B vs automatic", **pair(g["rater_b"], g["automatic_correct"])},
                 {"subset": label, "comparison": "rater consensus vs automatic",
                  **pair(consensus["rater_a"], consensus["automatic_correct"])}]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(folder, "human_validation_agreement.csv"), index=False)
    disputed = merged[(merged["rater_a"] == merged["rater_b"]) & (merged["rater_a"] != merged["automatic_correct"])]
    disputed.to_csv(os.path.join(folder, "items_where_both_raters_disagree_with_automatic_grade.csv"), index=False)
    print(table.to_string(index=False))
    print(f"\n{len(disputed)} items where both raters disagree with the automatic grade (listed in the CSV).")


if __name__ == "__main__":
    main()
