"""Builds a blinded sheet for an independent second rater.

All open-ended answers (root cause, recovery) from the main comparison are included, plus a random
sample of the remaining answers. Method names are hidden and rows are shuffled. The rater fills
'second_rater_correct' with 1 or 0 and saves the file as second_rater_sheet_completed.csv; the
analysis script then reports agreement and Cohen's kappa against the automatic grades.

Usage: python -m experiment_evaluation.prepare_second_rater_sheet --config configs/experiment_matrix.yaml --extra 60
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", default="main_comparison__qwen3_4b_instruct")
    parser.add_argument("--extra", type=int, default=60, help="additional non open-ended items to sample")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    questions = {}
    for path in glob.glob(os.path.join(cfg["benchmark_dir"], "*__test_questions.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                q = json.loads(line)
                questions[q["question_id"]] = q
    records = []
    with open(os.path.join(cfg["results_dir"], "raw_runs", f"{args.experiment}.jsonl"), "r", encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("status") == "ok":
                records.append(r)
    rng = random.Random(args.seed)
    open_ended = [r for r in records if questions[r["question_id"]].get("open_ended")]
    others = [r for r in records if not questions[r["question_id"]].get("open_ended")]
    chosen = open_ended + rng.sample(others, min(args.extra, len(others)))
    rng.shuffle(chosen)
    sheet, key = [], []
    for i, r in enumerate(chosen, start=1):
        q = questions[r["question_id"]]
        reference = q["gold"] if not q.get("keywords") else f'{q["gold"]} (acceptable: {"; ".join(q["keywords"])})'
        sheet.append({"item_id": i, "question": q["question"], "reference_answer": reference,
                      "system_answer": r["final_answer"], "second_rater_correct": ""})
        key.append({"item_id": i, "run_key": r["run_key"], "method": r["method"], "automatic_correct": int(r["correct"])})
    out_dir = os.path.join(cfg["results_dir"], "second_rater")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(sheet).to_csv(os.path.join(out_dir, "second_rater_sheet_blank.csv"), index=False)
    pd.DataFrame(key).to_csv(os.path.join(out_dir, "second_rater_answer_key.csv"), index=False)
    print(f"{len(sheet)} items written to {out_dir}/second_rater_sheet_blank.csv "
          "(give only that file to the rater; keep the answer key private).")


if __name__ == "__main__":
    main()
