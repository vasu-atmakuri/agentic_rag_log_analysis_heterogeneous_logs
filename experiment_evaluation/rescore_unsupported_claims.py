"""Recomputes the unsupported-claim fields of stored runs with the current checker version, without
calling any model. The original file is kept next to it with the suffix '.checker_v<old>'.

Usage: python -m experiment_evaluation.rescore_unsupported_claims --config configs/experiment_matrix_journal_extended.yaml
"""
import argparse
import json
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from experiment_evaluation.automatic_answer_grading import CHECKER_VERSION, UnsupportedClaimChecker  # noqa: E402
from log_analysis_system.log_analysis_workspace import LogAnalysisWorkspace  # noqa: E402
from log_analysis_system.log_format_detection_and_parsing import parse_log_file  # noqa: E402
from log_analysis_system.statistical_log_profiler import compute_log_statistics  # noqa: E402


def lightweight_workspace(ds, spec, artifacts_dir):
    """Parsed events only (no index, no model): all the claim checker needs."""
    import pickle

    cache = os.path.join(artifacts_dir, ds, "parsed_events_and_statistics.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as handle:
            c = pickle.load(handle)
        fmt, events, stats = c["format_name"], c["events"], c["stats"]
    else:
        parsed = parse_log_file(spec["log_path"], spec.get("base_year", 2005))
        fmt, events, stats = parsed.format_name, parsed.events, compute_log_statistics(parsed.events)
    return LogAnalysisWorkspace(ds, fmt, events, stats, {}, None, None, None, events, {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    datasets_cfg = yaml.safe_load(open(cfg["datasets_config"]))["datasets"]
    results = cfg["results_dir"] if args.split == "test" else os.path.join(cfg["results_dir"], "dev_smoke_tests")
    questions = {}
    for ds in datasets_cfg:
        with open(os.path.join(cfg["benchmark_dir"], f"{ds}__{args.split}_questions.jsonl")) as handle:
            for line in handle:
                q = json.loads(line)
                questions[q["question_id"]] = q
    checkers = {}
    raw_dir = os.path.join(results, "raw_runs")
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(raw_dir, name)
        records = [json.loads(line) for line in open(path) if line.strip()]
        old_versions = {r.get("claim_checker_version", 1) for r in records if r.get("status") == "ok"}
        if old_versions == {CHECKER_VERSION}:
            print(f"{name}: already at checker version {CHECKER_VERSION}")
            continue
        changed = 0
        for r in records:
            if r.get("status") != "ok":
                continue
            ds = r["dataset"]
            if ds not in checkers:
                checkers[ds] = UnsupportedClaimChecker(lightweight_workspace(ds, datasets_cfg[ds], cfg["artifacts_dir"]))
            before = r.get("has_unsupported_claim")
            r.update(checkers[ds].check(r.get("response_text", ""), questions[r["question_id"]]["question"]))
            r["claim_checker_version"] = CHECKER_VERSION
            changed += before != r["has_unsupported_claim"]
        backup = f"{path}.checker_v{max(old_versions) if old_versions else 1}"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
        with open(path, "w", encoding="utf-8") as handle:
            for r in records:
                handle.write(json.dumps(r) + "\n")
        print(f"{name}: rescored {len(records)} runs, {changed} flags changed (original kept as {os.path.basename(backup)})")


if __name__ == "__main__":
    main()
