"""Construction of the four complementary chunk types plus a fixed-size baseline chunking.

time_window     : non-overlapping 2-minute buckets, at most 50 events or 1,600 characters per chunk,
                  noisy loggers excluded.
error_burst     : ERROR/CRITICAL (error-like) events grouped into windows of 15 consecutive events,
                  sliding with a stride of 8 so that incident boundaries are not split.
logger_thread   : events of each important logger paginated into 1,600-character segments.
init_sequence   : lines containing initialization keywords within 5 minutes after a start marker.
fixed_size      : baseline mirroring common framework defaults: raw text split into 1,024-character
                  chunks with 128 characters of overlap, no metadata and no noise filtering.
"""
from __future__ import annotations

import re

from .log_format_detection_and_parsing import format_event
from .statistical_log_profiler import primary_events

TIME_WINDOW_SECONDS = 120
MAX_EVENTS_PER_WINDOW = 50
MAX_CHUNK_CHARS = 1600
ERROR_WINDOW_SIZE = 15
ERROR_WINDOW_STRIDE = 8
INIT_WINDOW_SECONDS = 300
FIXED_CHUNK_CHARS = 1024
FIXED_CHUNK_OVERLAP = 128
LINE_CHARS = 300

CHUNK_TYPES = ("time_window", "error_burst", "logger_thread", "init_sequence")
INIT_KEYWORDS = re.compile(r"\b(init\w*|start\w*|load\w*|version|model|config\w*|connect\w*)\b", re.IGNORECASE)
START_MARKER = re.compile(
    r"\b(starting|started|start up|startup|booting|boot|initializ\w+|launch\w*|listening on|bootstrap\w*|"
    r"service up|server started|establishing)\b", re.IGNORECASE)


def _make_chunk(chunk_type: str, events: list[dict], extra_header: str = "") -> dict:
    lines = [format_event(e, LINE_CHARS) for e in events]
    timed = [e for e in events if e["epoch"] is not None]
    start = timed[0] if timed else events[0]
    end = timed[-1] if timed else events[-1]
    loggers = sorted({e["logger"] for e in events})
    header = f"[{chunk_type}] {start['ts_str']} to {end['ts_str']}" + (f" | {extra_header}" if extra_header else "")
    return {
        "chunk_type": chunk_type, "start_ts": start["ts_str"], "end_ts": end["ts_str"],
        "start_epoch": start["epoch"], "end_epoch": end["epoch"], "loggers": loggers[:20],
        "line_numbers": [e["line_no"] for e in events], "text": header + "\n" + "\n".join(lines),
    }


def _pack(events: list[dict], max_events: int, max_chars: int):
    group, size = [], 0
    for e in events:
        length = min(len(e["raw"]), LINE_CHARS) + 1
        if group and (len(group) >= max_events or size + length > max_chars):
            yield group
            group, size = [], 0
        group.append(e)
        size += length
    if group:
        yield group


def build_time_window_chunks(events: list[dict], noisy_loggers: set[str]) -> list[dict]:
    chunks = []
    buckets: dict[int, list[dict]] = {}
    for e in events:
        if e["epoch"] is None or e["logger"] in noisy_loggers:
            continue
        buckets.setdefault(int(e["epoch"] // TIME_WINDOW_SECONDS), []).append(e)
    for key in sorted(buckets):
        for group in _pack(buckets[key], MAX_EVENTS_PER_WINDOW, MAX_CHUNK_CHARS):
            chunks.append(_make_chunk("time_window", group))
    return chunks


def build_error_burst_chunks(events: list[dict]) -> list[dict]:
    errors = [e for e in primary_events(events) if e["error_like"]]
    chunks = []
    if not errors:
        return chunks
    start = 0
    while True:
        window = errors[start:start + ERROR_WINDOW_SIZE]
        if not window:
            break
        chunks.append(_make_chunk("error_burst", window))
        if start + ERROR_WINDOW_SIZE >= len(errors):
            break
        start += ERROR_WINDOW_STRIDE
    return chunks


def build_logger_thread_chunks(events: list[dict], important_loggers: list[str]) -> list[dict]:
    chunks = []
    for logger in important_loggers:
        own = [e for e in events if e["logger"] == logger]
        for group in _pack(own, 10 ** 9, MAX_CHUNK_CHARS):
            chunks.append(_make_chunk("logger_thread", group, extra_header=f"logger={logger}"))
    return chunks


def build_init_sequence_chunks(events: list[dict]) -> list[dict]:
    primary = primary_events(events)
    markers = [e for e in primary if e["epoch"] is not None and START_MARKER.search(e["message"])]
    chunks, covered_until = [], None
    for marker in markers:
        if covered_until is not None and marker["epoch"] <= covered_until:
            continue
        window_end = marker["epoch"] + INIT_WINDOW_SECONDS
        selected = [e for e in primary if e["epoch"] is not None and marker["epoch"] <= e["epoch"] <= window_end
                    and INIT_KEYWORDS.search(e["message"])]
        covered_until = window_end
        for group in _pack(selected, 10 ** 9, MAX_CHUNK_CHARS):
            chunks.append(_make_chunk("init_sequence", group))
    return chunks


def build_multi_type_chunks(events: list[dict], important_loggers: list[str], noisy_loggers: list[str]) -> list[dict]:
    noisy = set(noisy_loggers)
    chunks = (build_time_window_chunks(events, noisy) + build_error_burst_chunks(events)
              + build_logger_thread_chunks(events, [lg for lg in important_loggers if lg not in noisy])
              + build_init_sequence_chunks(events))
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i
    return chunks


def build_fixed_size_chunks(events: list[dict]) -> list[dict]:
    chunks, buffer, buffer_lines = [], "", []
    for e in events:
        line = e["raw"][:LINE_CHARS] + "\n"
        buffer += line
        buffer_lines.append(e["line_no"])
        while len(buffer) >= FIXED_CHUNK_CHARS:
            text = buffer[:FIXED_CHUNK_CHARS]
            consumed = text.count("\n")
            chunks.append({"chunk_type": "fixed_size", "text": text, "line_numbers": buffer_lines[:max(1, consumed)],
                           "start_ts": "", "end_ts": "", "start_epoch": None, "end_epoch": None, "loggers": []})
            buffer = buffer[FIXED_CHUNK_CHARS - FIXED_CHUNK_OVERLAP:]
            kept = buffer.count("\n")
            buffer_lines = buffer_lines[len(buffer_lines) - kept:] if kept else []
    if buffer.strip():
        chunks.append({"chunk_type": "fixed_size", "text": buffer, "line_numbers": buffer_lines,
                       "start_ts": "", "end_ts": "", "start_epoch": None, "end_epoch": None, "loggers": []})
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i
    return chunks
