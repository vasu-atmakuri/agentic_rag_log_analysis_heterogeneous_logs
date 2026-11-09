"""Converts the RCAEval RE3 fault cases (Pham et al., 2025; Zenodo 14504481) into one plain-text log per case.

RE3 contains code-level faults injected into three microservice systems (Online Boutique, Sock Shop, Train Ticket),
each repeated three or four times. Every case directory holds logs.csv (all containers, nanosecond timestamps),
inject_time.txt (Unix seconds) and is named <root_cause_service>_<fault_id>/<run>/.

For each case this script writes
  log_datasets/rcaeval_re3_cases/<case_id>/<case_id>.log               one line per record, time ordered:
                                                                        <ISO-8601 UTC> <container> <message>
                                                                        (multi-line messages keep their line breaks)
  log_datasets/rcaeval_re3_cases/<case_id>/<case_id>_ground_truth.json  root-cause service, fault id, injection time
and the dataset list configs/benchmark_datasets_rcaeval_re3.yaml. The log content is not filtered or sampled.
Six cases (two per system, fixed seed) form the development split; all others are test cases.

Usage: python -m benchmark_construction.prepare_rcaeval_re3_root_cause_cases
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from datetime import datetime, timezone

import pandas as pd

SYSTEMS = {"RE3-OB": ("ob", "Online Boutique"), "RE3-SS": ("ss", "Sock Shop"), "RE3-TT": ("tt", "Train Ticket")}
SEED = 20261104
DEV_CASES_PER_SYSTEM = 2


def iso_utc(ns: int) -> str:
    seconds, rest = divmod(int(ns), 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{rest // 1000:06d}Z"


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def convert_case(case_dir: str, out_root: str) -> dict:
    system_dir, fault_dir, run = case_dir.rstrip("/").split("/")[-3:]
    sys_code, sys_name = SYSTEMS[system_dir]
    root_service, fault_id = fault_dir.rsplit("_", 1)
    case_id = f"rcaeval_re3_{sys_code}_{slug(root_service)}_{fault_id}_run{run}"
    out_dir = os.path.join(out_root, case_id)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{case_id}.log")
    truth_path = os.path.join(out_dir, f"{case_id}_ground_truth.json")

    df = pd.read_csv(os.path.join(case_dir, "logs.csv"), usecols=["timestamp", "container_name", "message"],
                     dtype={"container_name": str, "message": str}, keep_default_na=False)
    df = df[(df["message"].str.strip() != "") & (df["message"] != "nan") & (df["container_name"] != "")]
    df = df.sort_values("timestamp", kind="stable")
    with open(log_path, "w", encoding="utf-8") as handle:
        for ts, container, message in zip(df["timestamp"], df["container_name"], df["message"]):
            text = message.replace("\r\n", "\n").rstrip("\n")
            handle.write(f"{iso_utc(ts)} {container} {text}\n")

    inject = int(open(os.path.join(case_dir, "inject_time.txt"), encoding="utf-8").read().strip())
    containers = sorted(df["container_name"].unique())
    truth = {
        "case_id": case_id, "source": "RCAEval RE3 (Zenodo 14504481)", "system": sys_name, "system_code": sys_code,
        "original_case_path": f"{system_dir}/{fault_dir}/{run}", "root_cause_service": root_service,
        "fault_id": fault_id, "run": int(run), "inject_epoch": inject,
        "inject_utc": datetime.fromtimestamp(inject, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "log_start_utc": iso_utc(int(df["timestamp"].iloc[0])), "log_end_utc": iso_utc(int(df["timestamp"].iloc[-1])),
        "records": int(len(df)), "containers": containers,
        "root_cause_service_in_logs": root_service in containers,
    }
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump(truth, handle, indent=2)
    truth["log_path"], truth["ground_truth_path"] = log_path, truth_path
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--re3-dir", default="log_datasets/rcaeval/RE3")
    parser.add_argument("--out-root", default="log_datasets/rcaeval_re3_cases")
    parser.add_argument("--config-out", default="configs/benchmark_datasets_rcaeval_re3.yaml")
    args = parser.parse_args()

    case_dirs = sorted(glob.glob(os.path.join(args.re3_dir, "RE3-*", "*", "*", "")))
    case_dirs = [d for d in case_dirs if os.path.exists(os.path.join(d, "logs.csv"))]
    cases = []
    for d in case_dirs:
        truth = convert_case(d, args.out_root)
        cases.append(truth)
        print(f'{truth["case_id"]}: {truth["records"]} records, root cause {truth["root_cause_service"]}'
              f'{"" if truth["root_cause_service_in_logs"] else "  (NOT IN LOGS)"}')

    rng = random.Random(SEED)
    dev = set()
    for code in sorted({c["system_code"] for c in cases}):
        ids = sorted(c["case_id"] for c in cases if c["system_code"] == code)
        dev.update(rng.sample(ids, min(DEV_CASES_PER_SYSTEM, len(ids) - 1)))

    lines = [
        "# RCAEval RE3 root-cause localization cases (generated by prepare_rcaeval_re3_root_cause_cases.py).",
        "# One log source per case; each test case contributes one question with the labelled root-cause service.",
        f"seed: {SEED}",
        "benchmark_output_dir: benchmark_questions_frozen_rcaeval_re3",
        "",
        "datasets:",
    ]
    for c in sorted(cases, key=lambda c: c["case_id"]):
        lines += [f'  {c["case_id"]}:', "    reader: rcaeval_re3_case", f'    log_path: {c["log_path"]}',
                  f'    ground_truth: {c["ground_truth_path"]}', f'    system: {c["system_code"]}', "    time_style: iso",
                  f'    split: {"dev" if c["case_id"] in dev else "test"}']
    with open(args.config_out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{len(cases)} cases ({len(cases) - len(dev)} test, {len(dev)} dev); dataset list written to {args.config_out}")


if __name__ == "__main__":
    main()
