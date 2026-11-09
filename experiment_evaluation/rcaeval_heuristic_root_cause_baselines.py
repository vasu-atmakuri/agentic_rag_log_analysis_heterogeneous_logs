"""Non-model reference baselines for the RCAEval RE3 root-cause questions.

Each baseline ranks the containers of a case using only the parsed log and the injection time, comparing a window
after the injection with a window of equal length before it:
  error_count_increase   rise in error lines (ERROR/CRITICAL level, error wording when no level is logged, or
                         exception wording such as Traceback, Exception, panic or a *Error class name)
  log_volume_change      largest absolute log-ratio change in the number of events
  service_restart_signal rise in start-up lines (started, starting, initializing, listening on, ...) in the
                         five minutes after the injection compared with the five minutes before; added after
                         the model runs, when traces showed that RE3 code faults are deployed by restarting the
                         target service (study deviation 4)
  uniform_random         expected accuracy of picking a container at random
Reported metrics: AC@1 and AC@3 (share of cases whose root cause is ranked first / within the top three).

Usage: python -m experiment_evaluation.rcaeval_heuristic_root_cause_baselines --config configs/experiment_matrix_rcaeval_re3.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import re

import pandas as pd
import yaml

from log_analysis_system.log_format_detection_and_parsing import parse_log_file

STARTUP_WORDING = re.compile(r"\b(started|starting|start(ing)? the|listening on|initializ(ing|ed)|bootstrap|"
                             r"tomcat started|server started|application started)\b", re.IGNORECASE)
RESTART_WINDOW_SECONDS = 300
EXCEPTION_WORDING = re.compile(r"\b(traceback|exception|panic|fatal)\b|\w+Error\b", re.IGNORECASE)


def rank_case(events: list[dict], inject: float) -> dict[str, list[str]]:
    starts = [e["epoch"] for e in events if e["epoch"] is not None]
    window = min(inject - min(starts), max(starts) - inject)
    pre_err, post_err, pre_all, post_all = Counter(), Counter(), Counter(), Counter()
    for e in events:
        t = e["epoch"]
        if t is None or not (inject - window <= t < inject + window):
            continue
        is_error = e["error_like"] or bool(EXCEPTION_WORDING.search(e["message"]))
        after = t >= inject
        if e["level"] != "RAW":
            (post_all if after else pre_all)[e["logger"]] += 1
        if is_error:
            (post_err if after else pre_err)[e["logger"]] += 1
    pre_start, post_start = Counter(), Counter()
    for e in events:
        t = e["epoch"]
        if t is None or e["level"] == "RAW" or not STARTUP_WORDING.search(e["message"]):
            continue
        if inject - RESTART_WINDOW_SECONDS <= t < inject:
            pre_start[e["logger"]] += 1
        elif inject <= t < inject + RESTART_WINDOW_SECONDS:
            post_start[e["logger"]] += 1
    names = sorted(set(pre_all) | set(post_all))
    by_error = sorted(names, key=lambda n: (-(post_err[n] - pre_err[n]), n))
    by_volume = sorted(names, key=lambda n: (-abs(math.log((post_all[n] + 1) / (pre_all[n] + 1))), n))
    by_restart = sorted(names, key=lambda n: (-(post_start[n] - pre_start[n]), n))
    return {"error_count_increase": by_error, "log_volume_change": by_volume, "service_restart_signal": by_restart}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    datasets = yaml.safe_load(open(cfg["datasets_config"], encoding="utf-8"))["datasets"]
    out_dir = os.path.join(cfg["results_dir"], "analysis")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for name, spec in datasets.items():
        if spec["split"] != args.split:
            continue
        truth = json.load(open(spec["ground_truth"], encoding="utf-8"))
        events = parse_log_file(spec["log_path"]).events
        rankings = rank_case(events, float(truth["inject_epoch"]))
        for baseline, ranking in rankings.items():
            position = ranking.index(truth["root_cause_service"]) + 1 if truth["root_cause_service"] in ranking else None
            rows.append({"dataset": name, "system": truth["system_code"], "fault_id": truth["fault_id"],
                         "baseline": baseline, "gold": truth["root_cause_service"], "top3": ";".join(ranking[:3]),
                         "rank_of_root_cause": position, "ac1": int(position == 1), "ac3": int(position is not None and position <= 3)})
        n = len(truth["containers"])
        rows.append({"dataset": name, "system": truth["system_code"], "fault_id": truth["fault_id"],
                     "baseline": "uniform_random", "gold": truth["root_cause_service"], "top3": "",
                     "rank_of_root_cause": None, "ac1": 1 / n, "ac3": min(1.0, 3 / n)})
        print(f'{name}: error-increase top3 {rankings["error_count_increase"][:3]} (gold {truth["root_cause_service"]})')

    per_case = pd.DataFrame(rows)
    per_case.to_csv(os.path.join(out_dir, "rcaeval_heuristic_baselines_per_case.csv"), index=False)
    summary = (per_case.groupby(["baseline", "system"])[["ac1", "ac3"]].mean().unstack("system").round(3))
    overall = per_case.groupby("baseline")[["ac1", "ac3"]].mean().round(3)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.join(overall.rename(columns={"ac1": "ac1_all", "ac3": "ac3_all"}))
    summary.to_csv(os.path.join(out_dir, "rcaeval_heuristic_baselines_summary.csv"))
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
