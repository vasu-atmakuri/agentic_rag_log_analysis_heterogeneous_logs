"""Moves run records whose timing was distorted by the Mac sleeping (or the model server going down) out of
raw_runs/, so that the runner repeats them. Run only while no experiment is running.

A record is moved if suspended_seconds > 5 (recorded by the runner since 2026-11-05: wall-clock time minus
monotonic time, i.e. time spent asleep), or, for older records without that field, if wall_seconds > 600
(no method exceeds this when the machine is awake). Moved records are appended, unchanged, to
<results_dir>/archived_records_host_sleep/<experiment>.jsonl.

Usage: python scripts/archive_runs_interrupted_by_host_sleep.py experiment_results_journal_extended
"""
import glob
import json
import os
import sys


def interrupted(record: dict) -> bool:
    if record.get("status") != "ok":
        return False
    if "suspended_seconds" in record:
        return record["suspended_seconds"] > 5
    return (record.get("wall_seconds") or 0) > 600


def main(results_dir: str):
    archive = os.path.join(results_dir, "archived_records_host_sleep")
    total = 0
    for path in sorted(glob.glob(os.path.join(results_dir, "raw_runs", "*.jsonl"))):
        lines = open(path, encoding="utf-8").read().splitlines()
        keep, moved = [], []
        for line in lines:
            try:
                (moved if interrupted(json.loads(line)) else keep).append(line)
            except json.JSONDecodeError:
                keep.append(line)
        if moved:
            os.makedirs(archive, exist_ok=True)
            with open(os.path.join(archive, os.path.basename(path)), "a", encoding="utf-8") as handle:
                handle.write("\n".join(moved) + "\n")
            with open(path + ".tmp", "w", encoding="utf-8") as handle:
                handle.write("\n".join(keep) + ("\n" if keep else ""))
            os.replace(path + ".tmp", path)
            print(f"{os.path.basename(path)}: {len(moved)} records moved to {archive}")
            total += len(moved)
    print(f"{total} records archived; they will be repeated on the next run.")


if __name__ == "__main__":
    main(sys.argv[1])
