"""Shows, per experiment, how many runs are done, failed, and remaining, and whether a run is active.

Usage: python scripts/check_experiment_progress.py [--split dev]
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_matrix_journal_extended.yaml")
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    datasets_cfg = yaml.safe_load(open(cfg["datasets_config"]))["datasets"]
    results = cfg["results_dir"] if args.split == "test" else os.path.join(cfg["results_dir"], "dev_smoke_tests")
    running = subprocess.run(["pgrep", "-fl", "run_experiment_matrix"], capture_output=True, text=True).stdout.strip()
    print("Experiment process:", "RUNNING" if running else "not running")
    total_left = 0
    for exp in cfg["experiments"]:
        names = list(datasets_cfg) if exp.get("datasets", "all") == "all" else exp["datasets"]
        planned = 0
        for ds in names:
            suffix = f'__{exp["question_variant"]}' if exp.get("question_variant") else ""
            path = os.path.join(cfg["benchmark_dir"], f"{ds}__{args.split}_questions{suffix}.jsonl")
            n = sum(1 for _ in open(path)) if os.path.exists(path) else 0
            if exp.get("max_questions_per_dataset"):
                n = min(n, exp["max_questions_per_dataset"])
            planned += n * len(exp["methods"]) * int(exp.get("repeats", 1))
        ok, errors, last = set(), set(), None
        path = os.path.join(results, "raw_runs", f'{exp["name"]}.jsonl')
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                (ok if r.get("status") == "ok" else errors).add(r["run_key"])
                last = r.get("started_utc", last)
            errors -= ok
        left = max(0, planned - len(ok))
        total_left += left
        age = ""
        if last:
            minutes = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 60
            age = f"  last run started {minutes:.0f} min ago"
        print(f'{exp["name"][:62]:62} done {len(ok):4}/{planned:<4} failed {len(errors):3} left {left:4}{age}')
    print(f"Runs left in total: {total_left}")


if __name__ == "__main__":
    main()
