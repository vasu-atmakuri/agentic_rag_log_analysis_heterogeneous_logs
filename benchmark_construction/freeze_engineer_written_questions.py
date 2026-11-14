"""Converts the two engineer-written question sheets (first submitted versions) into frozen benchmark files.

Each answer was re-derived from the logs by the authors before freezing (all 50 confirmed). Answer types map to
the automatic grader as follows: name -> entity (writer's alternatives as aliases), count -> integer,
time -> timestamp (gold time of the event, 60 s tolerance as for template questions), short text -> keyword_any
(writer's answer and alternatives), and two-way comparisons -> choice_first_mention (the option named first in
the final answer counts as the model's choice). Six pairs of questions ask for the same fact from the two
writers; they are kept and flagged with `same_fact_as`.

Usage: python -m benchmark_construction.freeze_engineer_written_questions
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

SHEETS = {"A": "engineer_written_questions/completed/engineer_written_questions_writer_a.xlsx",
          "B": "engineer_written_questions/completed/engineer_written_questions_writer_b.xlsx"}
OUT_DIR = "benchmark_questions_frozen_engineer_written"
DATASET_OF_LOG = {"Hadoop_mapreduce.log": "loghub2_hadoop", "Zookeeper.log": "loghub2_zookeeper",
                  "Linux_syslog.log": "loghub2_linux", "Apache_error.log": "loghub2_apache",
                  "OpenSSH_first_250k_lines.log": "loghub2_openssh"}

# Time of the logged event behind each time question, read from the log line given by the writer (UTC as logged).
EVENT_TIMES = {"A03": "2015-10-17 21:48:25", "A08": "2015-07-29 17:42:28", "A13": "2005-07-25 23:23:13",
               "A17": "2005-06-09 06:07:19", "A22": "2015-12-10 09:32:20", "B01": "2015-10-17 21:49:49",
               "B08": "2015-07-29 17:42:28", "B12": "2005-06-09 06:06:41", "B25": "2015-12-10 07:13:56"}

# Two-way comparisons: option wordings, gold option first.
COMPARISON_OPTIONS = {
    "A04": [["spill", "spills", "spilling map output", "map-output spill", "map output spill"],
            ["commit", "commits", "committing", "task-commit", "task commit"]],
    "A09": [["follower", "followers", "following", "follower-election"],
            ["leader-election completions", "leading", "leader"]],
    "A14": [["sshd", "pam", "authentication failure", "authentication failures", "authentication-failure"],
            ["xinetd", "client address", "client-address"]],
    "A18": [["missing file", "missing files", "file does not exist", "missing"],
            ["forbidden", "directory index", "directory indexes"]],
    "A24": [["root"], ["invalid user", "invalid users", "invalid"]],
    "B04": [["maptask", "map task", "map"], ["reducetask", "reduce task", "reduce"]],
    "B07": [["3", "peer 3", "server 3"], ["2", "peer 2", "server 2"]],
    "B18": [["file does not exist", "missing file", "missing files"], ["directory index forbidden", "forbidden"]],
    "B24": [["root", "root account"], ["invalid user", "invalid users", "invalid"]],
}
SAME_FACT = {"A01": "B02", "A11": "B15", "A16": "B17", "A18": "B18", "A21": "B23", "A24": "B24"}
SAME_FACT.update({v: k for k, v in list(SAME_FACT.items())})


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def alternatives(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [a.strip() for a in str(value).split(";") if a.strip()]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    per_dataset = {d: [] for d in DATASET_OF_LOG.values()}
    for writer, path in SHEETS.items():
        sheet = pd.read_excel(path, sheet_name="Questions")
        for row in sheet.itertuples():
            qid, kind = row.question_id, str(row.answer_type).strip()
            gold = str(row.answer).strip()
            ds = DATASET_OF_LOG[row.log_file]
            q = {"question_id": f"engineer_{qid}", "dataset": ds, "split": "test", "writer": writer,
                 "category": row.category, "template": f"engineer_written_{row.category}", "question": row.question,
                 "gold": gold, "key_evidence_line_numbers": [], "how_found_by_writer": row.how_found,
                 "same_fact_as": f"engineer_{SAME_FACT[qid]}" if qid in SAME_FACT else None}
            if qid in COMPARISON_OPTIONS:
                q.update(answer_type="choice_first_mention", option_groups=COMPARISON_OPTIONS[qid])
            elif kind == "name":
                q.update(answer_type="entity", candidates=[], gold_aliases=alternatives(row.acceptable_alternatives))
            elif kind == "count":
                q.update(answer_type="integer", gold=int(float(gold)))
            elif kind == "time":
                epoch = datetime.strptime(EVENT_TIMES[qid], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
                q.update(answer_type="timestamp", gold_epoch=epoch, tolerance_seconds=60)
            elif kind == "short text":
                q.update(answer_type="keyword_any", keywords=[gold] + alternatives(row.acceptable_alternatives),
                         open_ended=True)
            else:
                raise ValueError(f"{qid}: unknown answer type {kind}")
            per_dataset[ds].append(q)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_sheets": {w: {"path": p, "sha256": sha256(p)} for w, p in SHEETS.items()}, "datasets": {}}
    for ds, items in per_dataset.items():
        for split in ("test", "dev"):
            path = os.path.join(OUT_DIR, f"{ds}__{split}_questions.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                for q in (items if split == "test" else []):
                    handle.write(json.dumps(q) + "\n")
        manifest["datasets"][ds] = {"test_questions": len(items),
                                    "test_questions_sha256": sha256(os.path.join(OUT_DIR, f"{ds}__test_questions.jsonl"))}
    with open(os.path.join(OUT_DIR, "benchmark_freeze_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print({ds: len(v) for ds, v in per_dataset.items()}, "total", sum(len(v) for v in per_dataset.values()))


if __name__ == "__main__":
    main()
