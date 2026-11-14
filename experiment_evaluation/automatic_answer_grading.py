"""Deterministic grading of final answers against programmatic gold answers, and an automatic
check of whether the concrete claims in a response are supported by the log.

Grading is applied only to the text after 'FINAL ANSWER:' (or the last line if absent).
Answer types: integer, numeric, entity, choice, timestamp, hour, keyword_any.

Unsupported-claim check: timestamps, dotted or CamelCase identifiers, IP addresses, version strings
and quoted strings found anywhere in the response are verified against the raw log text (and, for
timestamps, against event times within 60 seconds). Claims that also appear in the question are
ignored. A response is flagged as containing an unsupported claim if any claim fails.
"""
from __future__ import annotations

import bisect
import re
from datetime import datetime, timezone

TIMESTAMP_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2}(?:[.,]\d{1,6})?)?"),
    re.compile(r"\b[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{2}:\d{2}(?::\d{2})?"),
]
CLOCK_PATTERN = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_entity_text(text: str) -> str:
    text = text.lower().replace("$", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9().:\- ]+", " ", text)
    text = re.sub(r"(?<=[a-z0-9])[.:](?=[a-z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_timestamps(text: str) -> str:
    for pattern in TIMESTAMP_PATTERNS:
        text = pattern.sub(" ", text)
    return CLOCK_PATTERN.sub(" ", text)


EXTRA_TIMESTAMP_PATTERNS = [
    (re.compile(r"\b(\d{8})-(\d{1,2}):(\d{1,2}):(\d{1,2})(?::(\d{1,3}))?"), "compact"),
    (re.compile(r"\b[A-Z][a-z]{2} ([A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}:\d{2} \d{4})"), "%b %d %H:%M:%S %Y"),
    (re.compile(r"(?<![\d.])(\d{2}\.\d{2} \d{2}:\d{2}(?::\d{2})?)(?![\d.])"), "dotted"),
]


def _parse_extra_timestamp(text: str, gold_year: int):
    for pattern, kind in EXTRA_TIMESTAMP_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        try:
            if kind == "compact":
                day = datetime.strptime(m.group(1), "%Y%m%d")
                dt = day.replace(hour=int(m.group(2)), minute=int(m.group(3)), second=int(m.group(4)),
                                 microsecond=int(m.group(5) or 0) * 1000)
            elif kind == "dotted":
                raw = m.group(1)
                fmt = "%m.%d %H:%M:%S" if raw.count(":") == 2 else "%m.%d %H:%M"
                dt = datetime.strptime(raw, fmt).replace(year=gold_year)
            else:
                dt = datetime.strptime(m.group(1), kind)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def _parse_any_timestamp(text: str, gold_year: int):
    extra = _parse_extra_timestamp(text, gold_year)
    if extra is not None:
        return extra
    for pattern in TIMESTAMP_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(0).replace("T", " ").replace(",", " " if pattern is TIMESTAMP_PATTERNS[1] else ",")
            raw = re.sub(r"\s+", " ", raw).strip()
            for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                        "%b %d %H:%M:%S", "%b %d %H:%M"):
                try:
                    parsed = datetime.strptime(raw, fmt)
                except ValueError:
                    continue
                if parsed.year == 1900:
                    parsed = parsed.replace(year=gold_year)
                return parsed.replace(tzinfo=timezone.utc).timestamp()
    return None


def _entity_spans(answer_norm: str, names: list[str]) -> set[str]:
    spans = []
    for name in names:
        target = normalize_entity_text(name)
        if not target:
            continue
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(target) + r"(?![a-z0-9])", answer_norm):
            spans.append((m.start(), m.end(), name))
    maximal = {name for (s, e, name) in spans
               if not any((s2 <= s and e <= e2) and (e2 - s2) > (e - s) for (s2, e2, _) in spans)}
    return maximal


def grade_final_answer(question: dict, final_answer: str) -> dict:
    kind = question["answer_type"]
    answer = (final_answer or "").strip()
    gold = question["gold"]
    correct, detail = False, ""
    if kind == "integer":
        numbers = NUMBER_PATTERN.findall(_strip_timestamps(answer))
        value = None
        if numbers:
            try:
                value = int(float(numbers[0].replace(",", "")))
            except ValueError:
                value = None
        correct, detail = value == int(gold), f"parsed={value}"
    elif kind == "numeric":
        numbers = NUMBER_PATTERN.findall(_strip_timestamps(answer))
        value = float(numbers[0].replace(",", "")) if numbers else None
        correct = value is not None and abs(value - float(gold)) <= question.get("tolerance", 0.0) + 1e-9
        detail = f"parsed={value}"
    elif kind in ("entity", "choice"):
        names = list(question.get("candidates") or question.get("options") or [])
        aliases = question.get("gold_aliases") or []
        if gold not in names:
            names.append(gold)
        names.extend(aliases)
        found = _entity_spans(normalize_entity_text(answer), names)
        found = {gold if n in aliases else n for n in found}
        correct, detail = found == {gold}, f"mentioned={sorted(found)}"
    elif kind == "timestamp":
        gold_epoch = question["gold_epoch"]
        gold_dt = datetime.fromtimestamp(gold_epoch, tz=timezone.utc)
        value = _parse_any_timestamp(answer, gold_dt.year)
        if value is not None:
            correct = abs(value - gold_epoch) <= question.get("tolerance_seconds", 60)
            detail = f"parsed_epoch={value}"
        else:
            clock = CLOCK_PATTERN.search(answer)
            if clock:
                h, m, s = int(clock.group(1)), int(clock.group(2)), int(clock.group(3) or 0)
                candidate = gold_dt.replace(hour=h, minute=m, second=s, microsecond=0).timestamp()
                correct = abs(candidate - gold_epoch) <= question.get("tolerance_seconds", 60)
                detail = f"clock_only={h:02d}:{m:02d}:{s:02d}"
    elif kind == "hour":
        gold_hour = datetime.fromtimestamp(question["gold_epoch"], tz=timezone.utc).hour
        value = _parse_any_timestamp(answer, datetime.fromtimestamp(question["gold_epoch"], tz=timezone.utc).year)
        if value is not None:
            hour = datetime.fromtimestamp(value, tz=timezone.utc).hour
        else:
            clock = CLOCK_PATTERN.search(answer)
            hour = int(clock.group(1)) if clock else None
            if hour is None:
                m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", answer.lower())
                if m:
                    hour = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
        correct, detail = hour == gold_hour, f"parsed_hour={hour}"
    elif kind == "choice_first_mention":
        # two-way comparisons written by engineers: option_groups[0] holds the gold option and its wordings;
        # the option mentioned first in the final answer is taken as the model's choice
        normalized = normalize_entity_text(answer)
        first = None
        for group_index, group in enumerate(question["option_groups"]):
            for wording in group:
                target = normalize_entity_text(wording)
                m = re.search(r"(?<![a-z0-9])" + re.escape(target) + r"(?![a-z0-9])", normalized) if target else None
                if m and (first is None or m.start() < first[0]):
                    first = (m.start(), group_index)
        correct = first is not None and first[1] == 0
        detail = f"first_option={None if first is None else first[1]}"
    elif kind == "keyword_any":
        lowered = answer.lower()
        hits = [k for k in question.get("keywords", [gold]) if k.lower() in lowered]
        correct, detail = bool(hits), f"matched={hits}"
    else:
        raise ValueError(f"unknown answer type {kind}")
    return {"correct": bool(correct), "grading_detail": detail}


# --------------------------------------------------------------------------- unsupported claims
IDENTIFIER_DOTTED = re.compile(r"\b[A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)+\b")
IDENTIFIER_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+(?:\$\w+)?\b")
IP_ADDRESS = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
VERSION = re.compile(r"\bv\d+(?:\.\d+){1,3}\b")
QUOTED = re.compile(r"[\"'`]([^\"'`\n]{12,120})[\"'`]")
IGNORED_IDENTIFIERS = {"e.g", "i.e", "etc.", "a.m", "p.m"}


CHECKER_VERSION = 2
# Version 2 (applied to all runs): the reference text also includes each event in the normalized form the
# system itself shows to the model ("<timestamp> <LEVEL> <component>: <message>") and the chunk-type names
# that appear in excerpt headers. Version 1 flagged correct quotations of those normalized lines as unsupported.


class UnsupportedClaimChecker:
    def __init__(self, workspace):
        from log_analysis_system.log_format_detection_and_parsing import format_event
        from log_analysis_system.multi_type_chunk_builder import CHUNK_TYPES

        parts = [e["raw"] for e in workspace.events]
        parts += [format_event(e, 10 ** 6) for e in workspace.events]
        parts += list(CHUNK_TYPES) + ["fixed_size"]
        self.text_lower = re.sub(r"[ \t]+", " ", "\n".join(parts).lower())
        self.epochs = sorted(e["epoch"] for e in workspace.primary if e["epoch"] is not None)
        self.year = datetime.fromtimestamp(self.epochs[0], tz=timezone.utc).year if self.epochs else 2000
        self.workspace = workspace

    def _time_supported(self, epoch: float) -> bool:
        i = bisect.bisect_left(self.epochs, epoch - 60)
        return i < len(self.epochs) and self.epochs[i] <= epoch + 60

    def check(self, response_text: str, question_text: str) -> dict:
        text = re.sub(r"<think>.*?</think>", "", response_text or "", flags=re.DOTALL)
        question_lower = question_text.lower()
        claims, unsupported = [], []

        for pattern in TIMESTAMP_PATTERNS:
            for match in pattern.finditer(text):
                token = match.group(0)
                if token.lower() in question_lower:
                    continue
                epoch = self.workspace.resolve_user_time(re.sub(r"[T,]", lambda m: " " if m.group(0) == "T" else ",", token))
                if epoch is None:
                    continue
                claims.append(("timestamp", token))
                if not self._time_supported(epoch):
                    unsupported.append(("timestamp", token))
        candidates = set(IDENTIFIER_DOTTED.findall(text)) | set(IDENTIFIER_CAMEL.findall(text))
        candidates |= set(IP_ADDRESS.findall(text)) | set(VERSION.findall(text))
        for token in candidates:
            low = token.lower().strip(".")
            if low in IGNORED_IDENTIFIERS or len(low) < 4 or low in question_lower:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)+", low) and not IP_ADDRESS.fullmatch(token):
                continue
            kind = "ip" if IP_ADDRESS.fullmatch(token) else "version" if VERSION.fullmatch(token) else "identifier"
            claims.append((kind, token))
            if low not in self.text_lower:
                unsupported.append((kind, token))
        for quoted in QUOTED.findall(text):
            low = re.sub(r"[ \t]+", " ", quoted.lower().strip().rstrip(",.;"))
            if low in question_lower:
                continue
            claims.append(("quoted", quoted))
            if low not in self.text_lower:
                unsupported.append(("quoted", quoted))
        return {"claims_checked": len(claims), "unsupported_claims": len(unsupported),
                "unsupported_examples": [f"{k}:{v}" for k, v in unsupported[:5]],
                "has_unsupported_claim": bool(unsupported)}
