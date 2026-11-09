"""Prepared, cached state for one log source under one characterization condition:
parsed events, statistics, characterization, multi-type and fixed-size indexes, BM25 over lines."""
from __future__ import annotations

import json
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from rank_bm25 import BM25Okapi

from .log_format_detection_and_parsing import parse_log_file, parse_timestamp
from .statistical_log_profiler import compute_log_statistics, primary_events
from .log_source_characterization import (automatic_characterization, effective_important_loggers,
                                          expert_characterization, generic_characterization)
from .multi_type_chunk_builder import build_fixed_size_chunks, build_multi_type_chunks
from .semantic_chunk_index import SemanticChunkIndex, get_embedder

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize_for_bm25(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


@dataclass
class LogAnalysisWorkspace:
    dataset_name: str
    format_name: str
    events: list
    stats: dict
    characterization: dict
    multi_type_index: SemanticChunkIndex
    fixed_size_index: SemanticChunkIndex
    bm25: BM25Okapi
    bm25_line_events: list
    timing: dict

    @property
    def primary(self) -> list:
        return primary_events(self.events)

    def resolve_user_time(self, text: str):
        """Parse a time string supplied by a model or a question; yearless formats are placed
        in the year that falls inside the log's time range."""
        if not text:
            return None
        text = str(text).strip()
        if self.format_name in ("syslog_rfc3164", "proxifier_bracketed"):
            # Syslog lines record no year, so a year supplied in a query cannot be matched against the
            # log; it is ignored and the month, day and time are placed within the log's own span.
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}:\d{2}(?::\d{2})?)", text)
            if m:
                month = datetime(2000, int(m.group(2)), 1).strftime("%b")
                text = f"{month} {int(m.group(3))} {m.group(4)}"
            else:
                text = re.sub(r"^([A-Z][a-z]{2}\s+\d{1,2}),?\s+\d{4}\s+", r"\1 ", text)
                text = re.sub(r"(\d{2}:\d{2}(?::\d{2})?)\s+\d{4}$", r"\1", text)
        epochs = [e["epoch"] for e in self.primary if e["epoch"] is not None]
        if not epochs:
            return parse_timestamp(text)
        lo, hi = min(epochs), max(epochs)
        first_year = datetime.fromtimestamp(lo, tz=timezone.utc).year
        last_year = datetime.fromtimestamp(hi, tz=timezone.utc).year
        candidates = []
        for year in range(first_year, last_year + 1):
            value = parse_timestamp(text, default_year=year)
            if value is not None:
                candidates.append(value)
        if not candidates:
            return None
        inside = [c for c in candidates if lo - 86400 <= c <= hi + 86400]
        return inside[0] if inside else candidates[0]


def _cache_dir(root: str, dataset: str, condition_key: str) -> str:
    return os.path.join(root, dataset, condition_key)


def prepare_workspace(dataset_name: str, log_path: str, artifacts_root: str, characterization_condition: str,
                      backend=None, expert_config_path: str | None = None, embedder_kind: str = "sentence_transformer",
                      base_year_for_yearless: int = 2005, characterization_model_tag: str = "") -> LogAnalysisWorkspace:
    import time

    timing = {}
    base_dir = os.path.join(artifacts_root, dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    parsed_cache = os.path.join(base_dir, "parsed_events_and_statistics.pkl")
    if os.path.exists(parsed_cache):
        with open(parsed_cache, "rb") as handle:
            cached = pickle.load(handle)
        format_name, events, stats = cached["format_name"], cached["events"], cached["stats"]
        timing.update(cached.get("timing", {}))
    else:
        t0 = time.perf_counter()
        parsed = parse_log_file(log_path, base_year_for_yearless)
        t1 = time.perf_counter()
        stats = compute_log_statistics(parsed.events)
        t2 = time.perf_counter()
        format_name, events = parsed.format_name, parsed.events
        timing = {"parse_seconds": round(t1 - t0, 3), "statistics_seconds": round(t2 - t1, 3),
                  "events_per_second_parsing": round(len(events) / max(t1 - t0, 1e-9), 1),
                  "detection_scores": parsed.detection_scores}
        with open(parsed_cache, "wb") as handle:
            pickle.dump({"format_name": format_name, "events": events, "stats": stats, "timing": timing}, handle)

    condition_key = characterization_condition
    if characterization_condition == "automatic":
        from .log_source_characterization import CHARACTERIZATION_VERSION

        condition_key = (f"automatic_v{CHARACTERIZATION_VERSION}__"
                         f"{re.sub(r'[^A-Za-z0-9.]+', '_', characterization_model_tag)}")
    cdir = _cache_dir(artifacts_root, dataset_name, condition_key)
    os.makedirs(cdir, exist_ok=True)
    char_path = os.path.join(cdir, "characterization.json")
    if os.path.exists(char_path):
        with open(char_path, "r", encoding="utf-8") as handle:
            characterization = json.load(handle)
    else:
        if characterization_condition == "automatic":
            if backend is None:
                raise ValueError("Automatic characterization requires a model backend")
            characterization = automatic_characterization(backend, events, stats)
        elif characterization_condition == "generic":
            characterization = generic_characterization(stats)
        elif characterization_condition == "expert":
            characterization = expert_characterization(expert_config_path, stats)
        else:
            raise ValueError(characterization_condition)
        with open(char_path, "w", encoding="utf-8") as handle:
            json.dump(characterization, handle, indent=2)

    embedder = get_embedder(embedder_kind)
    multi_dir = os.path.join(cdir, f"multi_type_index__{embedder.name.split('/')[-1]}")
    if os.path.exists(os.path.join(multi_dir, "chunks.jsonl")):
        multi_index = SemanticChunkIndex.load(multi_dir, embedder)
    else:
        t0 = time.perf_counter()
        chunks = build_multi_type_chunks(events, effective_important_loggers(characterization, stats),
                                         characterization.get("noisy_loggers", []))
        multi_index = SemanticChunkIndex.build(chunks, embedder)
        multi_index.save(multi_dir)
        with open(os.path.join(multi_dir, "index_build_timing.json"), "w") as handle:
            json.dump({"chunks": len(chunks), "seconds": round(time.perf_counter() - t0, 3)}, handle)

    fixed_dir = os.path.join(base_dir, f"fixed_size_index__{embedder.name.split('/')[-1]}")
    if os.path.exists(os.path.join(fixed_dir, "chunks.jsonl")):
        fixed_index = SemanticChunkIndex.load(fixed_dir, embedder)
    else:
        fixed_index = SemanticChunkIndex.build(build_fixed_size_chunks(events), embedder)
        fixed_index.save(fixed_dir)

    line_events = [e for e in events]
    bm25 = BM25Okapi([tokenize_for_bm25(e["raw"]) or ["_"] for e in line_events])
    return LogAnalysisWorkspace(dataset_name, format_name, events, stats, characterization, multi_index,
                                fixed_index, bm25, line_events, timing)
