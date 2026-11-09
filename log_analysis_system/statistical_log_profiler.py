"""Deterministic statistical profiling of parsed log events (no language model involved)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

TOP_TEMPLATES = 30
BURST_MULTIPLIER = 5.0
SAMPLES_PER_LOGGER = 5
MAX_TOTAL_SAMPLES = 60

_TEMPLATE_RULES = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{12,}\b"), "<HEX>"),
    (re.compile(r"(?:/[\w.\-]+){2,}/?"), "<PATH>"),
    (re.compile(r"\b[\w.\-]+@[\w.\-]+\b"), "<ADDR>"),
    (re.compile(r"\d+(?:\.\d+)?"), "<NUM>"),
]


def message_template(message: str) -> str:
    text = message
    for pattern, token in _TEMPLATE_RULES:
        text = pattern.sub(token, text)
    return re.sub(r"\s+", " ", text).strip()[:200]


def hour_key(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:00")


def primary_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e["level"] != "RAW" or e.get("parent") is None]


def compute_log_statistics(events: list[dict]) -> dict:
    primary = primary_events(events)
    level_counts = Counter(e["level"] for e in primary)
    logger_totals = Counter(e["logger"] for e in primary)
    logger_errors = Counter(e["logger"] for e in primary if e["error_like"])
    logger_warns = Counter(e["logger"] for e in primary if e["level"] == "WARN")

    hourly_events: dict[str, int] = defaultdict(int)
    hourly_errors: dict[str, int] = defaultdict(int)
    for e in primary:
        if e["epoch"] is None:
            continue
        key = hour_key(e["epoch"])
        hourly_events[key] += 1
        if e["error_like"]:
            hourly_errors[key] += 1

    active_hours = len(hourly_events) or 1
    mean_errors = sum(hourly_errors.values()) / active_hours
    bursts = sorted(
        [{"hour": h, "errors": c} for h, c in hourly_errors.items() if mean_errors > 0 and c > BURST_MULTIPLIER * mean_errors],
        key=lambda x: x["hour"],
    )

    template_counts = Counter(message_template(e["message"]) for e in primary)
    samples: dict[str, list[str]] = defaultdict(list)
    total_samples = 0
    for e in primary:
        if not e["error_like"] or total_samples >= MAX_TOTAL_SAMPLES:
            continue
        if len(samples[e["logger"]]) < SAMPLES_PER_LOGGER:
            samples[e["logger"]].append(f'{e["ts_str"]} {e["level"]} {e["message"][:200]}')
            total_samples += 1

    epochs = [e["epoch"] for e in primary if e["epoch"] is not None]
    timed = [e for e in primary if e["epoch"] is not None]
    return {
        "total_events": len(primary),
        "total_lines": len(events),
        "continuation_lines": len(events) - len(primary),
        "first_timestamp": timed[0]["ts_str"] if timed else None,
        "last_timestamp": timed[-1]["ts_str"] if timed else None,
        "time_span_hours": round((max(epochs) - min(epochs)) / 3600, 2) if epochs else None,
        "level_distribution": dict(level_counts.most_common()),
        "logger_statistics": {
            name: {"events": total, "error_like": logger_errors[name], "warn": logger_warns[name],
                   "error_rate": round(logger_errors[name] / total, 4)}
            for name, total in logger_totals.most_common()
        },
        "hourly_event_counts": dict(sorted(hourly_events.items())),
        "hourly_error_counts": dict(sorted(hourly_errors.items())),
        "mean_errors_per_active_hour": round(mean_errors, 3),
        "error_burst_hours": bursts,
        "top_message_templates": [{"template": t, "count": c} for t, c in template_counts.most_common(TOP_TEMPLATES)],
        "sample_errors_by_logger": dict(samples),
    }


def compact_statistics_summary(stats: dict, max_loggers: int = 25, max_templates: int = 15) -> str:
    lines = [
        f'Total events: {stats["total_events"]} (plus {stats["continuation_lines"]} continuation lines)',
        f'Time range: {stats["first_timestamp"]} to {stats["last_timestamp"]} ({stats["time_span_hours"]} hours)',
        "Level distribution: " + ", ".join(f"{k}={v}" for k, v in stats["level_distribution"].items()),
        "Loggers (events / error-like / error rate):",
    ]
    for name, s in list(stats["logger_statistics"].items())[:max_loggers]:
        lines.append(f'  {name}: {s["events"]} / {s["error_like"]} / {s["error_rate"]:.3f}')
    if stats["error_burst_hours"]:
        lines.append("Error burst hours: " + ", ".join(f'{b["hour"]} ({b["errors"]})' for b in stats["error_burst_hours"][:10]))
    lines.append("Most frequent message templates:")
    for t in stats["top_message_templates"][:max_templates]:
        lines.append(f'  [{t["count"]}] {t["template"][:150]}')
    lines.append("Sample error messages:")
    for name, msgs in list(stats["sample_errors_by_logger"].items())[:10]:
        for msg in msgs[:2]:
            lines.append(f"  {name}: {msg[:160]}")
    return "\n".join(lines)
