"""Builds the blinded human-validation sheets for the extended study (two independent raters, same items).

Items are drawn from the main comparisons of the extended study and from the RCAEval RE3 runs, stratified by
log source and by the automatic grade (half graded correct, half graded incorrect where the stratum allows), so
that both kinds of grader error can be observed. Model and method names and the automatic grade are hidden;
rows are shuffled with a fixed seed. Each rater receives an identical sheet and records 1 (correct) or 0
(incorrect) for every item, without seeing the other rater's sheet.

Outputs in <extended results_dir>/human_validation/:
  human_validation_sheet_rater_a.xlsx / _rater_b.xlsx   (and .csv copies) give these to the raters
  human_validation_answer_key.csv                         keep private: item -> run, method, model, automatic grade

Usage: python -m experiment_evaluation.prepare_human_validation_sheets \
           --extended-config configs/experiment_matrix_journal_extended.yaml \
           --rcaeval-config configs/experiment_matrix_rcaeval_re3.yaml --extended-items 150 --rcaeval-items 50
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict

import pandas as pd
import yaml

INSTRUCTIONS = [
    "Purpose: check whether the automatic grading of answers about system logs is correct.",
    "For each row, compare the SYSTEM ANSWER with the REFERENCE ANSWER and enter 1 in rater_correct if the system "
    "answer gives the reference answer, otherwise 0. Use the notes column for anything unclear.",
    "Rules:",
    "1. Judge only the final answer given. Explanations around it do not matter unless they change the answer.",
    "2. Names (components, services, versions, hosts): correct only if the answer names the reference and no other "
    "candidate as the answer. Small spelling or case differences are fine.",
    "3. Times: correct if the answer gives the reference time to the minute (a difference under 60 seconds is fine; "
    "a different date or hour is wrong). For 'which hour' questions, the start of the correct hour is required.",
    "4. Counts and numbers: the exact number is required unless a tolerance is shown next to the reference.",
    "5. Open answers (root cause, recovery event): correct if the answer states the same failure or event as the "
    "reference, in any wording. Listing several possible causes without choosing one is incorrect.",
    "6. 'Unknown', 'insufficient evidence' or no answer is incorrect.",
    "Work alone and do not discuss items with the other rater until both sheets are returned.",
]


def load_questions(benchmark_dir: str) -> dict:
    questions = {}
    for path in glob.glob(os.path.join(benchmark_dir, "*__test_questions.jsonl")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                q = json.loads(line)
                questions[(q["dataset"], q["question_id"])] = q
    return questions


def load_records(results_dir: str, prefix: str) -> list[dict]:
    records = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "raw_runs", f"{prefix}*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                r.pop("trace", None)
                records[r["run_key"]] = r
    return list(records.values())


def reference_text(q: dict) -> str:
    text = str(q["gold"])
    if q.get("keywords"):
        text += f' (acceptable wording: {"; ".join(q["keywords"])})'
    if q.get("tolerance"):
        text += f' (tolerance: plus or minus {q["tolerance"]})'
    return text


def stratified(records: list[dict], total: int, stratum_of, rng: random.Random) -> list[dict]:
    groups = defaultdict(lambda: {True: [], False: []})
    for r in records:
        groups[stratum_of(r)][bool(r["correct"])].append(r)
    for g in groups.values():
        for items in g.values():
            rng.shuffle(items)
    chosen, keys = [], sorted(groups)
    if not keys:
        return []
    quota = {k: total // len(keys) + (1 if i < total % len(keys) else 0) for i, k in enumerate(keys)}
    for k in keys:
        want = quota[k]
        right, wrong = groups[k][True], groups[k][False]
        take_right = min(len(right), want // 2 + (want % 2))
        take_wrong = min(len(wrong), want - take_right)
        take_right = min(len(right), want - take_wrong)
        chosen += right[:take_right] + wrong[:take_wrong]
    leftovers = [r for k in keys for v in groups[k].values() for r in v if r not in chosen]
    rng.shuffle(leftovers)
    chosen += leftovers[: max(0, total - len(chosen))]
    return chosen


def write_xlsx(frame: pd.DataFrame, path: str):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    info = wb.active
    info.title = "Instructions"
    for i, line in enumerate(INSTRUCTIONS, start=1):
        cell = info.cell(row=i, column=1, value=line)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    info.column_dimensions["A"].width = 120
    sheet = wb.create_sheet("Items")
    headers = list(frame.columns)
    widths = {"item_id": 8, "question": 60, "reference_answer": 32, "system_answer": 60, "rater_correct": 14, "notes": 30}
    for c, h in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDDDDD")
        sheet.column_dimensions[cell.column_letter].width = widths.get(h, 20)
    for r, row in enumerate(frame.itertuples(index=False), start=2):
        for c, value in enumerate(row, start=1):
            cell = sheet.cell(row=r, column=c, value=value)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    rule = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
    sheet.add_data_validation(rule)
    col = sheet.cell(row=1, column=headers.index("rater_correct") + 1).column_letter
    rule.add(f"{col}2:{col}{len(frame) + 1}")
    sheet.freeze_panes = "A2"
    wb.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extended-config", required=True)
    parser.add_argument("--rcaeval-config", required=True)
    parser.add_argument("--extended-items", type=int, default=150)
    parser.add_argument("--rcaeval-items", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    ext = yaml.safe_load(open(args.extended_config, encoding="utf-8"))
    rca = yaml.safe_load(open(args.rcaeval_config, encoding="utf-8"))

    pools = []
    for cfg, prefix, n, stratum in ((ext, "main_comparison__", args.extended_items, lambda r: r["dataset"]),
                                    (rca, "rcaeval_root_cause__", args.rcaeval_items, lambda r: r["dataset"][12:14])):
        questions = load_questions(cfg["benchmark_dir"])
        records = [r for r in load_records(cfg["results_dir"], prefix) if (r["dataset"], r["question_id"]) in questions]
        picked = stratified(records, n, stratum, rng)
        pools += [(r, questions[(r["dataset"], r["question_id"])]) for r in picked]
        print(f"{prefix}: {len(picked)} items from {len(records)} graded runs")
    rng.shuffle(pools)

    rows, key = [], []
    for i, (r, q) in enumerate(pools, start=1):
        rows.append({"item_id": i, "question": q["question"], "reference_answer": reference_text(q),
                     "system_answer": str(r.get("final_answer") or "").strip() or "(no answer given)", "rater_correct": "", "notes": ""})
        key.append({"item_id": i, "run_key": r["run_key"], "experiment": r["experiment"], "dataset": r["dataset"],
                    "question_id": r["question_id"], "answer_type": q["answer_type"], "method": r["method"],
                    "model": r["model"], "automatic_correct": int(bool(r["correct"]))})
    out_dir = os.path.join(ext["results_dir"], "human_validation")
    os.makedirs(out_dir, exist_ok=True)
    frame = pd.DataFrame(rows)
    pd.DataFrame(key).to_csv(os.path.join(out_dir, "human_validation_answer_key.csv"), index=False)
    for rater in ("a", "b"):
        base = os.path.join(out_dir, f"human_validation_sheet_rater_{rater}")
        frame.to_csv(base + ".csv", index=False)
        try:
            write_xlsx(frame, base + ".xlsx")
        except ImportError:
            print("openpyxl not installed: CSV sheets only")
    print(f"{len(frame)} items; sheets and private answer key written to {out_dir}")


if __name__ == "__main__":
    main()
