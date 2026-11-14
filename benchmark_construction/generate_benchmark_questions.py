"""Template-based benchmark construction with programmatically computed gold answers.

Four categories: factual recall, temporal reasoning, comparative, causal. Candidate questions are
generated from every applicable template, filtered for unambiguous gold answers (unique maxima,
clear margins, no events at window boundaries), then sampled round-robin across templates with a
fixed seed. A disjoint development split is drawn for debugging; the test split is frozen with
SHA-256 hashes of the questions and of the source logs.

Usage: python -m benchmark_construction.generate_benchmark_questions --config configs/benchmark_datasets_journal_extended.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone

import yaml

from .reference_log_readers import read_fault_injected_testbed, read_linux_syslog, read_loghub2, read_zookeeper

CATEGORIES = ("factual", "temporal", "comparative", "causal")


# ---------------------------------------------------------------- time formatting
def make_time_formatter(style: str):
    def minute(epoch: float) -> str:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return dt.strftime("%b %d %H:%M") if style == "syslog" else dt.strftime("%Y-%m-%d %H:%M")

    def day(epoch: float) -> str:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return dt.strftime("%b %d") if style == "syslog" else dt.strftime("%Y-%m-%d")

    return minute, day


def hour_floor(epoch: float) -> float:
    return epoch - (epoch % 3600)


# ---------------------------------------------------------------- event classes
def level_class(level: str) -> dict:
    return {"name": f"level_{level.lower()}", "description": f"{level}-level events", "singular": f"{level}-level event",
            "match": lambda e, lv=level: e["level"] == lv}


def template_class(event_id: str, template: str) -> dict:
    """Event class defined by the Loghub-2.0 authors' template annotation (gold uses event_id, never text matching)."""
    shown = template if len(template) <= 90 else template[:87] + "..."
    return {"name": f"template_{event_id}", "description": f"log lines matching the pattern '{shown}'",
            "singular": f"log line matching the pattern '{shown}'",
            "match": lambda e, eid=event_id: e.get("event_id") == eid}


def select_template_classes(events, count: int, rng: random.Random, min_occurrences=20, max_occurrences=5000):
    import re as _re

    counts = Counter(e["event_id"] for e in events if e.get("event_id"))
    text = {e["event_id"]: e["event_template"] for e in events if e.get("event_id")}
    eligible = []
    for eid, n in counts.items():
        t = text[eid]
        constant = _re.sub(r"<\*>", "", t)
        if min_occurrences <= n <= max_occurrences and len(constant.strip()) >= 12 and len(t) <= 160:
            eligible.append(eid)
    eligible.sort()
    rng.shuffle(eligible)
    return [template_class(eid, text[eid]) for eid in eligible[:count]]


def keyword_class(keyword: str) -> dict:
    return {"name": "keyword_" + "".join(ch if ch.isalnum() else "_" for ch in keyword.lower()).strip("_"),
            "description": f"events whose message contains '{keyword}'",
            "singular": f"event whose message contains '{keyword}'",
            "match": lambda e, kw=keyword.lower(): kw in e["message"].lower()}


# ---------------------------------------------------------------- helpers
def entity_question(qid_seed, category, template, text, gold, candidates, evidence=None, aliases=None):
    return {"category": category, "template": template, "question": text, "answer_type": "entity",
            "gold": gold, "gold_aliases": aliases or [], "candidates": sorted(set(candidates)),
            "key_evidence_line_numbers": evidence or [], "_seed": qid_seed}


def simple_question(qid_seed, category, template, text, answer_type, gold, evidence=None, **grading):
    q = {"category": category, "template": template, "question": text, "answer_type": answer_type, "gold": gold,
         "key_evidence_line_numbers": evidence or [], "_seed": qid_seed}
    q.update(grading)
    return q


def unique_max(counter: Counter, margin: float = 0.10):
    ranked = counter.most_common(2)
    if not ranked or ranked[0][1] == 0:
        return None
    if len(ranked) == 1:
        return ranked[0][0]
    top, second = ranked
    return top[0] if top[1] > second[1] and (top[1] - second[1]) >= max(1, margin * top[1]) else None


def boundary_clear(events, matcher, t, before=1.0, after=1.0):
    """True when no matching event lies in [t - before, t + after), i.e. the boundary is unambiguous."""
    return not any(matcher(e) and t - before <= e["epoch"] < t + after for e in events)


def window_clear(events, matcher, t1, t2):
    # an event inside the end minute (e.g. 16:00:30 for 'until 16:00') is ambiguous
    return boundary_clear(events, matcher, t1) and boundary_clear(events, matcher, t2, before=1.0, after=60.0)


# ---------------------------------------------------------------- generic templates
def generic_candidates(events, classes, components, minute, day, rng, window_hours, ds):
    pools = defaultdict(list)
    t_min, t_max = events[0]["epoch"], events[-1]["epoch"]
    by_class = {c["name"]: [e for e in events if c["match"](e)] for c in classes}

    for c in classes:
        hits = by_class[c["name"]]
        if not hits:
            continue
        first = hits[0]
        if not components and len([e for e in hits if e["epoch"] == first["epoch"]]) == 1:
            pools[("factual", "first_time_of_class")].append(simple_question(
                f'{ds}:F0:{c["name"]}', "factual", "first_time_of_class",
                f'At what time was the first {c["singular"]} logged?', "timestamp", first["ts_str"],
                [first["line_no"]], gold_epoch=first["epoch"], tolerance_seconds=60))
        if components and len([e for e in hits if e["epoch"] == first["epoch"]]) == 1:
            pools[("factual", "first_component_of_class")].append(entity_question(
                f'{ds}:F1:{c["name"]}', "factual", "first_component_of_class",
                f'Which component logged the first {c["singular"]} in the log?', first["component"], components,
                [first["line_no"]]))
        per_component = defaultdict(list)
        for e in hits:
            per_component[e["component"]].append(e)
        for comp, comp_hits in (per_component.items() if components else []):
            pools[("factual", "first_time_of_class_for_component")].append(simple_question(
                f'{ds}:F2:{c["name"]}:{comp}', "factual", "first_time_of_class_for_component",
                f'At what time did {comp} log its first {c["singular"]}?', "timestamp", comp_hits[0]["ts_str"],
                [comp_hits[0]["line_no"]], gold_epoch=comp_hits[0]["epoch"], tolerance_seconds=60))
        # first after a time point, and the event right before it (precedence)
        for _ in range(12):
            t = hour_floor(rng.uniform(t_min, t_max))
            after = [e for e in hits if e["epoch"] > t]
            if not after or after[0] is first:
                continue
            target = after[0]
            same_time = [e for e in events if abs(e["epoch"] - target["epoch"]) < 1e-6]
            if components and len(same_time) == 1 and boundary_clear(events, c["match"], t):
                pools[("factual", "first_component_of_class_after_time")].append(entity_question(
                    f'{ds}:F3:{c["name"]}:{int(t)}', "factual", "first_component_of_class_after_time",
                    f'Which component logged the first {c["singular"]} after {minute(t)}?', target["component"],
                    components, [target["line_no"]]))
                idx = events.index(target)
                if idx > 0:
                    prev = events[idx - 1]
                    if prev["epoch"] < target["epoch"] and (idx < 2 or events[idx - 2]["epoch"] < prev["epoch"]):
                        pools[("causal", "component_immediately_preceding_first_class_event")].append(entity_question(
                            f'{ds}:P1:{c["name"]}:{int(t)}', "causal", "component_immediately_preceding_first_class_event",
                            f'Which component logged the event immediately before the first {c["singular"]} after {minute(t)}?',
                            prev["component"], components, [prev["line_no"], target["line_no"]]))

        # counts in windows
        for _ in range(40):
            hours = rng.choice(window_hours)
            t1 = hour_floor(rng.uniform(t_min, t_max))
            t2 = t1 + hours * 3600
            inside = [e for e in hits if t1 <= e["epoch"] <= t2]
            if 2 <= len(inside) <= 60 and window_clear(events, c["match"], t1, t2):
                pools[("temporal", "count_class_in_window")].append(simple_question(
                    f'{ds}:T1:{c["name"]}:{int(t1)}:{hours}', "temporal", "count_class_in_window",
                    f'How many {c["description"]} were logged between {minute(t1)} and {minute(t2)}?', "integer", len(inside)))
        # peak hour on a day
        per_day = defaultdict(Counter)
        for e in hits:
            per_day[day(e["epoch"])][hour_floor(e["epoch"])] += 1
        for d, counter in per_day.items():
            top = unique_max(counter, margin=0.0)
            if top is not None and counter[top] >= 2:
                pools[("temporal", "peak_hour_of_class_on_day")].append(simple_question(
                    f'{ds}:T3:{c["name"]}:{d}', "temporal", "peak_hour_of_class_on_day",
                    f'During which hour of {d} were the most {c["description"]} logged? Give the starting time of that hour.',
                    "hour", minute(top), gold_epoch=top))
        # comparatives
        counts = Counter(e["component"] for e in hits)
        top = unique_max(counts)
        if top and components:
            pools[("comparative", "top_component_for_class")].append(entity_question(
                f'{ds}:C2:{c["name"]}', "comparative", "top_component_for_class",
                f'Which component logged the most {c["description"]} in the whole log?', top, components))
        names = [n for n, v in counts.items() if v > 0]
        for _ in range(10):
            if len(components) < 2:
                break
            a, b = rng.sample(sorted(set(components)), 2)
            ca, cb = counts.get(a, 0), counts.get(b, 0)
            if max(ca, cb) == 0 or (a not in names and b not in names):
                continue
            if abs(ca - cb) >= max(1, 0.2 * max(ca, cb)):
                gold, other = (a, b) if ca > cb else (b, a)
                pools[("comparative", "which_of_two_components_more_class")].append(simple_question(
                    f'{ds}:C1:{c["name"]}:{a}:{b}', "comparative", "which_of_two_components_more_class",
                    f'Which logged more {c["description"]}: {a} or {b}?', "choice", gold, options=[gold, other]))

    # component activity in windows
    for _ in range(60):
        hours = rng.choice(window_hours)
        t1 = hour_floor(rng.uniform(t_min, t_max))
        t2 = t1 + hours * 3600
        inside = [e for e in events if t1 <= e["epoch"] <= t2]
        if not inside:
            continue
        if not components:
            break
        counts = Counter(e["component"] for e in inside)
        top = unique_max(counts)
        extended = Counter(e["component"] for e in events if t1 <= e["epoch"] < t2 + 60)
        if top and unique_max(extended) == top and boundary_clear(events, lambda e: True, t1):
            pools[("comparative", "top_component_in_window")].append(entity_question(
                f'{ds}:C3:{int(t1)}:{hours}', "comparative", "top_component_in_window",
                f'Which component logged the most events between {minute(t1)} and {minute(t2)}?', top, components))
        comp = rng.choice(sorted(counts))
        if 2 <= counts[comp] <= 80 and window_clear(events, lambda e, cp=comp: e["component"] == cp, t1, t2):
            pools[("temporal", "count_component_events_in_window")].append(simple_question(
                f'{ds}:T2:{comp}:{int(t1)}:{hours}', "temporal", "count_component_events_in_window",
                f'How many events did {comp} log between {minute(t1)} and {minute(t2)}?', "integer", counts[comp]))
    return pools


# ---------------------------------------------------------------- testbed-specific templates
def testbed_candidates(events, manifest, components, minute):
    pools = defaultdict(list)
    versions = sorted({s["model_version"] for s in manifest["service_starts"]})
    for s in manifest["service_starts"]:
        when = minute(datetime.strptime(s["at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
        start_line = [e for e in events if e.get("tag") == s["start_id"]][0]["line_no"]
        pools[("factual", "model_version_at_start")].append(entity_question(
            f'tb:V:{s["start_id"]}', "factual", "model_version_at_start",
            f"Which model version was loaded when the service started at {when}?", s["model_version"], versions, [start_line]))
        pools[("factual", "configured_batch_size_at_start")].append(simple_question(
            f'tb:B:{s["start_id"]}', "factual", "configured_batch_size_at_start",
            f"What batch_size was configured when the service started at {when}?", "integer", s["batch_size"], [start_line + 2]))
        pools[("factual", "scheduler_workers_at_start")].append(simple_question(
            f'tb:W:{s["start_id"]}', "factual", "scheduler_workers_at_start",
            f"How many workers did the scheduler start with when the service started at {when}?", "integer", s["workers"]))
    redeploy = manifest["service_starts"][1]
    pools[("factual", "model_load_duration")].append(simple_question(
        "tb:L:redeploy", "factual", "model_load_duration",
        f'How many seconds did it take to load model version {redeploy["model_version"]} during the redeploy?',
        "numeric", redeploy["load_seconds"], tolerance=0.05))
    hosts = [manifest["db_host"].split(":")[0], manifest["cache_host"].split(":")[0], "idp.internal", "prometheus-gw.internal"]
    pools[("factual", "infrastructure_identifiers")].append(entity_question(
        "tb:H:db", "factual", "infrastructure_identifiers", "Which database host does the service connect to?",
        manifest["db_host"].split(":")[0], hosts))
    pools[("factual", "infrastructure_identifiers")].append(entity_question(
        "tb:H:cache", "factual", "infrastructure_identifiers", "Which Redis host does the feature store use as its cache?",
        manifest["cache_host"].split(":")[0], hosts))
    pools[("factual", "infrastructure_identifiers")].append(simple_question(
        "tb:H:port", "factual", "infrastructure_identifiers", "On which port does the API gateway listen?", "integer",
        manifest["service_starts"][0]["port"]))
    for inc in manifest["incidents"]:
        if inc.get("used_for_causal_questions") is False:
            continue
        iid = inc["incident_id"]
        root = [e for e in events if e.get("tag") == f"{iid}:root"][0]
        symptom = [e for e in events if e.get("tag") == f"{iid}:symptom"][0]
        around = minute(symptom["epoch"] - symptom["epoch"] % 60)
        running = [s for s in manifest["service_starts"]
                   if datetime.strptime(s["at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() <= root["epoch"]][-1]
        pools[("factual", "model_version_during_incident")].append(entity_question(
            f"tb:VI:{iid}", "factual", "model_version_during_incident",
            f'Which model version was running during {inc["symptom"]} around {around}?', running["model_version"], versions))
        pools[("causal", "incident_root_component")].append(entity_question(
            f"tb:RC:{iid}", "causal", "incident_root_component",
            f'Which component first reported the problem that led to {inc["symptom"]} around {around}?',
            inc["root_logger"], components, [root["line_no"]]))
        pools[("causal", "incident_root_cause")].append(simple_question(
            f"tb:RF:{iid}", "causal", "incident_root_cause",
            f'What underlying failure caused {inc["symptom"]} around {around}?', "keyword_any",
            inc["root_cause_keywords"][0], [root["line_no"]], keywords=inc["root_cause_keywords"], open_ended=True))
        resolution = [e for e in events if e.get("tag") == f"{iid}:resolution"]
        if resolution:
            pools[("causal", "incident_resolution")].append(simple_question(
                f"tb:RS:{iid}", "causal", "incident_resolution",
                f'What event marked the recovery from {inc["symptom"]} around {around}?', "keyword_any",
                inc["resolution_keywords"][0], [resolution[0]["line_no"]], keywords=inc["resolution_keywords"], open_ended=True))
    return pools


# ---------------------------------------------------------------- selection and freezing
def select(pools, targets: dict, dev_count: int, rng: random.Random, dataset: str):
    test, dev = [], []
    for category in CATEGORIES:
        templates = sorted(k for k in pools if k[0] == category)
        queues = {}
        for key in templates:
            items = list({q["_seed"]: q for q in pools[key]}.values())  # deduplicate
            rng.shuffle(items)
            queues[key] = items
        chosen = []
        while len(chosen) < targets.get(category, 0) and any(queues.values()):
            for key in templates:
                if queues[key] and len(chosen) < targets[category]:
                    chosen.append(queues[key].pop())
        test.extend(chosen)
        leftovers = [q for key in templates for q in queues[key]]
        rng.shuffle(leftovers)
        dev.extend(leftovers[: max(1, dev_count // len(CATEGORIES))])
    for split, items in (("test", test), ("dev", dev)):
        for i, q in enumerate(items, start=1):
            q["question_id"] = f"{dataset}__{split}__{i:03d}"
            q["dataset"] = dataset
            q["split"] = split
            q.pop("_seed", None)
    return test, dev


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset_questions(name: str, spec: dict, seed: int):
    rng = random.Random(f"{seed}:{name}")
    style = spec.get("time_style", "iso")
    minute, day = make_time_formatter(style)
    if spec["reader"] == "fault_injected_testbed":
        events, manifest = read_fault_injected_testbed(spec["ground_truth_manifest"])
    elif spec["reader"] == "zookeeper":
        events, manifest = read_zookeeper(spec["log_path"]), None
    elif spec["reader"] == "linux_syslog":
        events, manifest = read_linux_syslog(spec["log_path"], spec.get("base_year", 2005)), None
    elif spec["reader"] == "loghub2":
        events, manifest = read_loghub2(spec["loghub2_name"], spec["log_path"], spec["structured_csv"],
                                        spec.get("max_lines")), None
    else:
        raise ValueError(spec["reader"])
    events.sort(key=lambda e: (e["epoch"], e["line_no"]))
    components = sorted({e["component"] for e in events})
    classes = [level_class(lv) for lv in spec.get("level_classes", [])] + [keyword_class(k) for k in spec.get("keyword_classes", [])]
    if spec.get("template_classes"):
        classes += select_template_classes(events, spec["template_classes"], random.Random(f"{seed}:{name}:templates"))
    if spec.get("no_component_questions"):
        components = []
    pools = generic_candidates(events, classes, components, minute, day, rng, spec.get("window_hours", [1, 2, 3]), name)
    if manifest is not None:
        for key, items in testbed_candidates(events, manifest, components, minute).items():
            pools[key].extend(items)
        if not spec.get("include_precedence_causal", False):
            pools.pop(("causal", "component_immediately_preceding_first_class_event"), None)
    test, dev = select(pools, spec["questions_per_category"], spec.get("dev_questions", 8), rng, name)
    pool_sizes = {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(pools.items())}
    return test, dev, pool_sizes, len(events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    out_dir = cfg["benchmark_output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    freeze = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": cfg["seed"], "datasets": {}}
    for name, spec in cfg["datasets"].items():
        test, dev, pool_sizes, n_events = build_dataset_questions(name, spec, cfg["seed"])
        for split, items in (("test", test), ("dev", dev)):
            path = os.path.join(out_dir, f"{name}__{split}_questions.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                for q in items:
                    handle.write(json.dumps(q) + "\n")
        test_path = os.path.join(out_dir, f"{name}__test_questions.jsonl")
        freeze["datasets"][name] = {
            "log_sha256": sha256_file(spec["log_path"]), "test_questions_sha256": sha256_file(test_path),
            "reference_events": n_events, "test_questions": len(test), "dev_questions": len(dev),
            "test_by_category": dict(Counter(q["category"] for q in test)),
            "test_by_template": dict(Counter(q["template"] for q in test)), "candidate_pool_sizes": pool_sizes,
        }
        print(f"{name}: {len(test)} test / {len(dev)} dev questions  {dict(Counter(q['category'] for q in test))}")
    with open(os.path.join(out_dir, "benchmark_freeze_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(freeze, handle, indent=2)
    print("Freeze manifest written. Post it (and the question files) before running any experiment.")


if __name__ == "__main__":
    main()
