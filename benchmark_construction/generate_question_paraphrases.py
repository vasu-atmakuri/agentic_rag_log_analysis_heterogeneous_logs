"""Two deterministic, rule-based rewordings of every test question (same gold answer, same grading fields).

Each template has a pattern that must match the original wording exactly; a question whose wording does not
match is reported and copied unchanged, so the coverage of the paraphrase sets is auditable.

Usage: python -m benchmark_construction.generate_question_paraphrases --benchmark-dir benchmark_questions_frozen_journal_extended
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

C = r"(?P<comp>.+?)"
T = r"(?P<t>.+?)"
T1, T2 = r"(?P<t1>.+?)", r"(?P<t2>.+?)"
S = r"(?P<s>.+)"
D = r"(?P<d>.+)"
X = r"(?P<x>.+?)"

RULES = {
    "first_component_of_class": (rf"^Which component logged the first {S} in the log\?$",
        "In the whole log, which component produced the earliest {s}?",
        "Looking at the entire log, what component emitted the very first {s}?"),
    "first_time_of_class_for_component": (rf"^At what time did {C} log its first {S}\?$",
        "When did {comp} first emit a {s}?",
        "What is the timestamp of the earliest {s} from {comp}?"),
    "first_time_of_class": (rf"^At what time was the first {S} logged\?$",
        "When was the earliest {s} recorded?",
        "Give the timestamp of the first {s} in the log."),
    "first_component_of_class_after_time": (rf"^Which component logged the first {S} after {T}\?$",
        "After {t}, which component was the first to log a {s}?",
        "Which component produced the earliest {s} following {t}?"),
    "component_immediately_preceding_first_class_event": (
        rf"^Which component logged the event immediately before the first {S} after {T}\?$",
        "Consider the first {s} after {t}. Which component logged the event right before it?",
        "Which component wrote the log entry directly preceding the earliest {s} after {t}?"),
    "count_class_in_window": (rf"^How many {D} were logged between {T1} and {T2}\?$",
        "Count the {d} recorded from {t1} to {t2}.",
        "What is the number of {d} between {t1} and {t2}?"),
    "count_component_events_in_window": (rf"^How many events did {C} log between {T1} and {T2}\?$",
        "Count the events logged by {comp} from {t1} to {t2}.",
        "What number of log entries did {comp} produce between {t1} and {t2}?"),
    "peak_hour_of_class_on_day": (
        rf"^During which hour of (?P<day>.+?) were the most {D} logged\? Give the starting time of that hour\.$",
        "On {day}, which hour had the highest number of {d}? Give the hour's starting time.",
        "Find the hour of {day} with the most {d} and state when that hour starts."),
    "top_component_for_class": (rf"^Which component logged the most {D} in the whole log\?$",
        "Across the entire log, which component produced the largest number of {d}?",
        "Which component has the highest count of {d} overall?"),
    "which_of_two_components_more_class": (rf"^Which logged more {D}: (?P<a>.+) or (?P<b>.+)\?$",
        "Between {a} and {b}, which one logged more {d}?",
        "Did {a} or {b} produce a larger number of {d}?"),
    "top_component_in_window": (rf"^Which component logged the most events between {T1} and {T2}\?$",
        "From {t1} to {t2}, which component was the most active, with the most events?",
        "Which component produced the largest number of log entries between {t1} and {t2}?"),
    "model_version_at_start": (rf"^Which model version was loaded when the service started at {T}\?$",
        "For the service start at {t}, which model version was loaded?",
        "What model version did the service load during its start at {t}?"),
    "configured_batch_size_at_start": (rf"^What batch_size was configured when the service started at {T}\?$",
        "For the service start at {t}, what was the configured batch_size?",
        "Which batch_size value was set in the configuration at the {t} service start?"),
    "scheduler_workers_at_start": (
        rf"^How many workers did the scheduler start with when the service started at {T}\?$",
        "At the {t} service start, how many workers did the scheduler start?",
        "What was the scheduler's worker count when the service started at {t}?"),
    "model_load_duration": (
        r"^How many seconds did it take to load model version (?P<v>\S+) during the redeploy\?$",
        "During the redeploy, how long in seconds did loading model version {v} take?",
        "What was the load time in seconds for model version {v} at the redeploy?"),
    "model_version_during_incident": (rf"^Which model version was running during {X} around {T}\?$",
        "Around {t}, during {x}, which model version was running?",
        "What model version was in service when {x} occurred around {t}?"),
    "incident_root_component": (
        rf"^Which component first reported the problem that led to {X} around {T}\?$",
        "Around {t}, {x} occurred. Which component reported the underlying problem first?",
        "Which component was the first to log the problem behind {x} around {t}?"),
    "incident_root_cause": (rf"^What underlying failure caused {X} around {T}\?$",
        "What was the root cause of {x} around {t}?",
        "Around {t}, {x} occurred. What failure caused it?"),
    "incident_resolution": (rf"^What event marked the recovery from {X} around {T}\?$",
        "Which log event shows that {x} around {t} was resolved?",
        "What event indicates recovery after {x} around {t}?"),
}
FIXED = {
    "Which database host does the service connect to?": (
        "What is the hostname of the database the service connects to?",
        "To which database host does the service open its connections?"),
    "Which Redis host does the feature store use as its cache?": (
        "What Redis host serves as the feature store's cache?",
        "Which host runs the Redis cache used by the feature store?"),
    "On which port does the API gateway listen?": (
        "What port is the API gateway listening on?",
        "Which port number does the API gateway accept requests on?"),
}


def paraphrase(question: dict) -> tuple[str, str, bool]:
    text = question["question"]
    if text in FIXED:
        return FIXED[text][0], FIXED[text][1], True
    rule = RULES.get(question["template"])
    if rule:
        m = re.match(rule[0], text)
        if m:
            g = m.groupdict()
            return rule[1].format(**g), rule[2].format(**g), True
    return text, text, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    total = unmatched = 0
    for path in sorted(glob.glob(os.path.join(args.benchmark_dir, f"*__{args.split}_questions.jsonl"))):
        questions = [json.loads(line) for line in open(path, encoding="utf-8")]
        outs = {1: [], 2: []}
        for q in questions:
            p1, p2, ok = paraphrase(q)
            total += 1
            if not ok:
                unmatched += 1
                print(f"  not paraphrased: {q['question_id']} ({q['template']})")
            for idx, text in ((1, p1), (2, p2)):
                outs[idx].append({**q, "question": text, "original_question": q["question"], "paraphrase_set": idx})
        for idx, rows in outs.items():
            out = path.replace(".jsonl", f"__paraphrase_{idx}.jsonl")
            with open(out, "w", encoding="utf-8") as handle:
                for r in rows:
                    handle.write(json.dumps(r) + "\n")
    print(f"{total} questions, {total - unmatched} paraphrased, {unmatched} copied unchanged")


if __name__ == "__main__":
    main()
