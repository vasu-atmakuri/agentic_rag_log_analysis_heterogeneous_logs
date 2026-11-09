"""Root-cause localization questions for the RCAEval RE3 cases.

Each case yields one question. The gold answer is the root-cause service named by the RCAEval authors (the fault
was injected into that service's code); it is never derived from the log text. The question gives the system and
the injection time to the minute, which is the information RCAEval also gives to the methods it evaluates.
Answer candidates are all containers that appear in the case's log, so the grader rejects answers that name more
than one service.

Usage: python -m benchmark_construction.generate_rcaeval_root_cause_questions --config configs/benchmark_datasets_rcaeval_re3.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone

import yaml

SYSTEM_NAMES = {"ob": "Online Boutique", "ss": "Sock Shop", "tt": "Train Ticket"}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def root_cause_question(name: str, truth: dict, split: str) -> dict:
    inject_minute = datetime.fromtimestamp(truth["inject_epoch"] - truth["inject_epoch"] % 60, tz=timezone.utc)
    system = SYSTEM_NAMES[truth["system_code"]]
    text = (f"In this {system} microservice deployment, a fault was injected into the code of one service at about "
            f"{inject_minute.strftime('%Y-%m-%d %H:%M')} UTC. Based on the logs, which service is the root cause of "
            f"the fault? Answer with the service name only.")
    return {"question_id": f"{name}__{split}__001", "dataset": name, "split": split, "category": "causal",
            "template": "root_cause_service_localization", "question": text, "answer_type": "entity",
            "gold": truth["root_cause_service"], "gold_aliases": [], "candidates": sorted(truth["containers"]),
            "key_evidence_line_numbers": [], "fault_id": truth["fault_id"], "system": truth["system_code"],
            "inject_epoch": truth["inject_epoch"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = cfg["benchmark_output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    freeze = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": cfg["seed"], "datasets": {}}
    per_split = Counter()
    for name, spec in cfg["datasets"].items():
        truth = json.load(open(spec["ground_truth"], encoding="utf-8"))
        if not truth["root_cause_service_in_logs"]:
            print(f"{name}: root-cause service absent from the logs; case excluded")
            continue
        question = root_cause_question(name, truth, spec["split"])
        for split in ("test", "dev"):
            path = os.path.join(out_dir, f"{name}__{split}_questions.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                if split == spec["split"]:
                    handle.write(json.dumps(question) + "\n")
        per_split[spec["split"]] += 1
        freeze["datasets"][name] = {
            "split": spec["split"], "log_sha256": sha256_file(spec["log_path"]),
            "test_questions_sha256": sha256_file(os.path.join(out_dir, f"{name}__test_questions.jsonl")),
            "root_cause_service": truth["root_cause_service"], "fault_id": truth["fault_id"],
            "system": truth["system_code"], "records": truth["records"], "candidates": len(truth["containers"]),
        }
    with open(os.path.join(out_dir, "benchmark_freeze_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(freeze, handle, indent=2)
    print(f"{per_split['test']} test and {per_split['dev']} dev root-cause questions written to {out_dir}")


if __name__ == "__main__":
    main()
