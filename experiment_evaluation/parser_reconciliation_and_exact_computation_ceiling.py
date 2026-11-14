"""Reconciles the system's log parser with the reference readers and measures the exact-computation ceiling.

For each Loghub-2.0 source it reports, for the parser revision selected with LOG_PARSER_REVISION (1: main study,
2: corrected sensitivity version):
  primary_events     share of lines the parser recognizes as events (others become continuation lines)
  timestamp_agree    share of reference events whose parser time equals the reference time (to the second)
  component_agree    share of reference events whose parser component equals the reference component
  ceiling            frozen test questions whose gold answer is reproduced when the generator's answer rule is
                     executed over the parser's view of the log (parser times and components; event class
                     membership taken from the Loghub annotation by line number), i.e. the accuracy of exact
                     computation with the correct query
and lists every question whose answer differs, with the reason.

Usage: LOG_PARSER_REVISION=1 python -m experiment_evaluation.parser_reconciliation_and_exact_computation_ceiling
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter

import yaml

from benchmark_construction.reference_log_readers import read_loghub2
from benchmark_construction.verify_gold_answers_with_corrected_readers import recompute

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "parser_module", os.path.join(HERE, "log_analysis_system", "log_format_detection_and_parsing.py"))
parser_module = importlib.util.module_from_spec(_spec)
sys.modules["parser_module"] = parser_module
_spec.loader.exec_module(parser_module)

CONFIG = "configs/benchmark_datasets_journal_extended.yaml"
BENCHMARK = "benchmark_questions_frozen_journal_extended"


def parser_view(spec: dict, reference: dict) -> list[dict]:
    parsed = parser_module.parse_log_file(spec["log_path"], spec.get("base_year", 2005))
    view = []
    for e in parsed.events:
        if e["level"] == "RAW" and e.get("parent") is not None:
            continue  # continuation line: not an event in the system's view
        ref = reference.get(e["line_no"])
        view.append({"line_no": e["line_no"], "epoch": e["epoch"], "ts_str": e["ts_str"],
                     "component": e["logger"], "level": e["level"] if e["level"] != "UNSPECIFIED" else (ref or {}).get("level", "UNSPECIFIED"),
                     "message": (ref or {}).get("message", e["message"]),
                     "event_id": (ref or {}).get("event_id"), "event_template": (ref or {}).get("event_template")})
    view.sort(key=lambda x: (x["epoch"] if x["epoch"] is not None else 0, x["line_no"]))
    return view, parsed


def main():
    revision = parser_module.PARSER_REVISION
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    total = Counter()
    print(f"parser revision {revision}")
    print("source format lines primary_events timestamp_agree component_agree ceiling")
    for name, spec in cfg["datasets"].items():
        if spec.get("reader") != "loghub2":
            continue
        ref_events = read_loghub2(spec["loghub2_name"], spec["log_path"], spec["structured_csv"], spec.get("max_lines"))
        reference = {e["line_no"]: e for e in ref_events}
        view, parsed = parser_view(spec, reference)
        by_line = {e["line_no"]: e for e in view}
        lines = len(parsed.events)
        ts_ok = sum(1 for e in ref_events if e["line_no"] in by_line and by_line[e["line_no"]]["epoch"] is not None
                    and abs(by_line[e["line_no"]]["epoch"] - e["epoch"]) < 1)
        has_comp = any(e["component"] for e in ref_events)
        comp_ok = sum(1 for e in ref_events if e["line_no"] in by_line and by_line[e["line_no"]]["component"] == e["component"])
        ref_sorted = sorted(ref_events, key=lambda e: (e["epoch"], e["line_no"]))
        same, diffs = 0, []
        for line in open(os.path.join(BENCHMARK, f"{name}__test_questions.jsonl"), encoding="utf-8"):
            q = json.loads(line)
            status, gold, extra = recompute(q, view, spec.get("time_style", "iso"))
            if status == "ok":
                ok = (abs(extra["gold_epoch"] - q["gold_epoch"]) < 60) if "gold_epoch" in extra and "gold_epoch" in q \
                    else str(gold) == str(q["gold"])
            else:
                ok = False
            if ok:
                same += 1
            else:
                ref_status, ref_gold, _ = recompute(q, ref_sorted, spec.get("time_style", "iso"))
                diffs.append({"question_id": q["question_id"], "template": q["template"], "status": status,
                              "parser_answer": gold, "gold": q["gold"], "reference_recomputed": ref_gold})
        n = sum(1 for _ in open(os.path.join(BENCHMARK, f"{name}__test_questions.jsonl"), encoding="utf-8"))
        total.update({"same": same, "questions": n})
        print(f"{name} {parsed.format_name} {lines} {len(view)/lines:.3f} {ts_ok/len(ref_events):.4f} "
              f"{(comp_ok/len(ref_events)) if has_comp else float('nan'):.4f} {same}/{n}")
        for d in diffs:
            print("   differs:", json.dumps(d, default=str))
    print(f"TOTAL ceiling {total['same']}/{total['questions']} = {100*total['same']/total['questions']:.1f}%")


if __name__ == "__main__":
    main()
