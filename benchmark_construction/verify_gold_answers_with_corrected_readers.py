"""Recomputes the gold answer of every frozen Loghub-2.0 test question with the corrected reference readers.

Version 1 of the readers skipped lines whose header did not match their pattern (HealthApp dates written without
zero padding, Mac process names containing spaces or a trailing bracketed context, HPC lines with a negative flag
field). This script reads each source again with the corrected readers, recovers each question's parameters from
its frozen wording, applies the same answer rules and ambiguity checks as the question generator, and classifies
every question as
  unchanged        same gold answer
  gold_changed     unambiguous, but the gold answer differs (the corrected answer is used)
  now_ambiguous    the corrected log makes the question ambiguous under the generator's rules (excluded)
  unresolved       parameters could not be recovered from the wording (reported, kept with version 1 gold)
It writes corrected question files to <benchmark_dir>_gold_v2/ (excluded questions are omitted there) and a
per-question report. The frozen version-1 files are not modified.

Usage: python -m benchmark_construction.verify_gold_answers_with_corrected_readers \
           --config configs/benchmark_datasets_journal_extended.yaml
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import yaml

from .generate_benchmark_questions import boundary_clear, hour_floor, make_time_formatter, unique_max, window_clear
from .reference_log_readers import READ_COVERAGE, read_fault_injected_testbed, read_loghub2


_TEMPLATES: dict = {}


def class_matcher(text: str, events: list[dict]):
    """Returns a predicate for the event class named in a question, or None if it cannot be resolved."""
    m = re.match(r"^(?:log lines? matching the pattern) '(?P<t>.*)'$", text)
    if m:
        shown = m["t"]
        key = id(events)
        if key not in _TEMPLATES:
            _TEMPLATES[key] = {e["event_id"]: e["event_template"] for e in events if e.get("event_id")}
        templates = _TEMPLATES[key]
        if shown.endswith("..."):
            ids = [i for i, t in templates.items() if len(t) > 90 and t.startswith(shown[:-3])]
        else:
            ids = [i for i, t in templates.items() if t == shown]
        if len(ids) != 1:
            return None
        return lambda e, eid=ids[0]: e.get("event_id") == eid
    m = re.match(r"^(?P<lv>[A-Z]+)-level events?$", text)
    if m:
        return lambda e, lv=m["lv"]: e["level"] == lv
    m = re.match(r"^events? whose message contains '(?P<k>.*)'$", text)
    if m:
        return lambda e, kw=m["k"].lower(): kw in e["message"].lower()
    return None


def time_parser(style: str, events: list[dict]):
    years = sorted({datetime.fromtimestamp(e["epoch"], tz=timezone.utc).year for e in events})
    lo, hi = events[0]["epoch"] - 86400 * 2, events[-1]["epoch"] + 86400 * 2

    def minute(text: str):
        if style != "syslog":
            return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()
        for y in years:
            t = datetime.strptime(f"{y} {text}", "%Y %b %d %H:%M").replace(tzinfo=timezone.utc).timestamp()
            if lo <= t <= hi:
                return t
        return None
    return minute


PATTERNS = {
    "first_component_of_class": r"^Which component logged the first (?P<s>.+) in the log\?$",
    "first_time_of_class": r"^At what time was the first (?P<s>.+) logged\?$",
    "first_time_of_class_for_component": r"^At what time did (?P<comp>.+?) log its first (?P<s>.+)\?$",
    "first_component_of_class_after_time": r"^Which component logged the first (?P<s>.+) after (?P<t>.+?)\?$",
    "component_immediately_preceding_first_class_event":
        r"^Which component logged the event immediately before the first (?P<s>.+) after (?P<t>.+?)\?$",
    "count_class_in_window": r"^How many (?P<d>.+) were logged between (?P<t1>.+?) and (?P<t2>.+?)\?$",
    "count_component_events_in_window": r"^How many events did (?P<comp>.+?) log between (?P<t1>.+?) and (?P<t2>.+?)\?$",
    "peak_hour_of_class_on_day":
        r"^During which hour of (?P<day>.+?) were the most (?P<d>.+) logged\? Give the starting time of that hour\.$",
    "top_component_for_class": r"^Which component logged the most (?P<d>.+) in the whole log\?$",
    "which_of_two_components_more_class": r"^Which logged more (?P<d>.+): (?P<a>.+) or (?P<b>.+)\?$",
    "top_component_in_window": r"^Which component logged the most events between (?P<t1>.+?) and (?P<t2>.+?)\?$",
}


_HITS: dict = {}


def recompute(q: dict, events: list[dict], style: str):
    """Returns (status, new_gold, extra) for one question."""
    rule = PATTERNS.get(q["template"])
    m = re.match(rule, q["question"]) if rule else None
    if not m:
        return "unresolved", None, {}
    g = m.groupdict()
    to_epoch = time_parser(style, events)
    minute_fmt, day_fmt = make_time_formatter(style)
    match = class_matcher(g.get("s") or g.get("d") or "", events) if (g.get("s") or g.get("d")) else None
    if (g.get("s") or g.get("d")) and match is None:
        return "unresolved", None, {}
    tpl = q["template"]
    class_text = g.get("s") or g.get("d") or ""
    if match and (id(events), class_text) not in _HITS:
        _HITS[(id(events), class_text)] = [e for e in events if match(e)]
    hits = _HITS.get((id(events), class_text), []) if match else []

    if tpl in ("first_component_of_class", "first_time_of_class"):
        if not hits or sum(1 for e in hits if e["epoch"] == hits[0]["epoch"]) != 1:
            return "now_ambiguous", None, {}
        first = hits[0]
        if tpl == "first_component_of_class":
            return "ok", first["component"], {}
        return "ok", first["ts_str"], {"gold_epoch": first["epoch"]}
    if tpl == "first_time_of_class_for_component":
        mine = [e for e in hits if e["component"] == g["comp"]]
        if not mine:
            return "now_ambiguous", None, {}
        return "ok", mine[0]["ts_str"], {"gold_epoch": mine[0]["epoch"]}
    if tpl in ("first_component_of_class_after_time", "component_immediately_preceding_first_class_event"):
        t = to_epoch(g["t"])
        if t is None:
            return "unresolved", None, {}
        after = [e for e in hits if e["epoch"] > t]
        if not after or after[0] is hits[0] or not boundary_clear(events, match, t):
            return "now_ambiguous", None, {}
        target = after[0]
        if sum(1 for e in events if abs(e["epoch"] - target["epoch"]) < 1e-6) != 1:
            return "now_ambiguous", None, {}
        if tpl == "first_component_of_class_after_time":
            return "ok", target["component"], {}
        idx = events.index(target)
        if idx == 0:
            return "now_ambiguous", None, {}
        prev = events[idx - 1]
        if not (prev["epoch"] < target["epoch"] and (idx < 2 or events[idx - 2]["epoch"] < prev["epoch"])):
            return "now_ambiguous", None, {}
        return "ok", prev["component"], {}
    if tpl in ("count_class_in_window", "count_component_events_in_window", "top_component_in_window"):
        t1, t2 = to_epoch(g["t1"]), to_epoch(g["t2"])
        if t1 is None or t2 is None:
            return "unresolved", None, {}
        if tpl == "count_class_in_window":
            if not window_clear(events, match, t1, t2):
                return "now_ambiguous", None, {}
            return "ok", sum(1 for e in hits if t1 <= e["epoch"] <= t2), {}
        if tpl == "count_component_events_in_window":
            comp_match = lambda e, cp=g["comp"]: e["component"] == cp  # noqa: E731
            if not window_clear(events, comp_match, t1, t2):
                return "now_ambiguous", None, {}
            return "ok", sum(1 for e in events if comp_match(e) and t1 <= e["epoch"] <= t2), {}
        inside = Counter(e["component"] for e in events if t1 <= e["epoch"] <= t2)
        extended = Counter(e["component"] for e in events if t1 <= e["epoch"] < t2 + 60)
        top = unique_max(inside)
        if not top or unique_max(extended) != top or not boundary_clear(events, lambda e: True, t1):
            return "now_ambiguous", None, {}
        return "ok", top, {}
    if tpl == "peak_hour_of_class_on_day":
        per_hour = Counter(hour_floor(e["epoch"]) for e in hits if day_fmt(e["epoch"]) == g["day"])
        top = unique_max(per_hour, margin=0.0)
        if top is None or per_hour[top] < 2:
            return "now_ambiguous", None, {}
        return "ok", minute_fmt(top), {"gold_epoch": top}
    if tpl == "top_component_for_class":
        top = unique_max(Counter(e["component"] for e in hits))
        return ("ok", top, {}) if top else ("now_ambiguous", None, {})
    if tpl == "which_of_two_components_more_class":
        counts = Counter(e["component"] for e in hits)
        ca, cb = counts.get(g["a"], 0), counts.get(g["b"], 0)
        if max(ca, cb) == 0 or abs(ca - cb) < max(1, 0.2 * max(ca, cb)):
            return "now_ambiguous", None, {}
        return "ok", g["a"] if ca > cb else g["b"], {}
    return "unresolved", None, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    src_dir = cfg["benchmark_output_dir"]
    out_dir = src_dir + "_gold_v2"
    os.makedirs(out_dir, exist_ok=True)
    report, summary = [], defaultdict(Counter)
    for name, spec in cfg["datasets"].items():
        files = {split: os.path.join(src_dir, f"{name}__{split}_questions.jsonl") for split in ("test", "dev")}
        if spec["reader"] != "loghub2":
            for split, path in files.items():  # sources without a corrected reader are copied unchanged
                with open(path, encoding="utf-8") as handle, \
                        open(os.path.join(out_dir, os.path.basename(path)), "w", encoding="utf-8") as out:
                    out.write(handle.read())
            continue
        events = read_loghub2(spec["loghub2_name"], spec["log_path"], spec["structured_csv"], spec.get("max_lines"))
        events.sort(key=lambda e: (e["epoch"], e["line_no"]))
        coverage = READ_COVERAGE[spec["loghub2_name"]]
        for split, path in files.items():
            kept = []
            for line in open(path, encoding="utf-8"):
                q = json.loads(line)
                status, gold, extra = recompute(q, events, spec.get("time_style", "iso"))
                if status == "ok":
                    same = (str(gold) == str(q["gold"])) if "gold_epoch" not in extra else \
                        abs(extra["gold_epoch"] - q.get("gold_epoch", extra["gold_epoch"])) < 60
                    status = "unchanged" if same else "gold_changed"
                summary[name][f"{split}:{status}"] += 1
                report.append({"dataset": name, "split": split, "question_id": q["question_id"], "template": q["template"],
                               "status": status, "gold_v1": q["gold"], "gold_v2": gold if gold is not None else ""})
                if status == "now_ambiguous":
                    continue
                if status == "gold_changed":
                    q = {**q, "gold": gold, **extra, "gold_v1": q["gold"]}
                    if q["answer_type"] == "entity" and gold not in q.get("candidates", []):
                        q["candidates"] = sorted(set(q.get("candidates", [])) | {gold})
                    if q["answer_type"] == "choice":
                        q["options"] = [gold] + [o for o in q["options"] if o != gold]
                kept.append(q)
            with open(os.path.join(out_dir, os.path.basename(path)), "w", encoding="utf-8") as out:
                for q in kept:
                    out.write(json.dumps(q) + "\n")
        print(f"{name}: lines read {coverage['lines_read']}, events {coverage['events']}, skipped {coverage['skipped']}; "
              + ", ".join(f"{k}={v}" for k, v in sorted(summary[name].items())))
    with open(os.path.join(out_dir, "gold_v2_verification_report.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    totals = Counter()
    for c in summary.values():
        totals.update(c)
    print("TOTAL", dict(sorted(totals.items())))


if __name__ == "__main__":
    main()
