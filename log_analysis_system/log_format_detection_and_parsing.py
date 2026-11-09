"""Format-agnostic log parsing.

Format profiles are tried in priority order on a stratified sample of up to
300 lines (beginning, middle, end). The profile with the highest match rate is
selected, provided that rate exceeds 20 percent; otherwise every line becomes an
unstructured RAW event. Lines that do not match the selected profile (stack
traces, wrapped messages) are kept as continuation events with level RAW that
inherit timestamp and logger from their parent event.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

SAMPLE_LINES = 300
MATCH_RATE_THRESHOLD = 0.20

LEVEL_ALIASES = {
    "WARNING": "WARN", "ERR": "ERROR", "FATAL": "CRITICAL", "CRIT": "CRITICAL",
    "EMERG": "CRITICAL", "ALERT": "CRITICAL", "NOTICE": "INFO", "TRACE": "DEBUG",
}
KNOWN_LEVELS = {"DEBUG", "INFO", "WARN", "ERROR", "CRITICAL", "UNSPECIFIED", "RAW"}

# Heuristic error marker for formats without a severity field (syslog).
ERROR_LIKE_PATTERN = re.compile(
    r"\b(fail(?:ed|ure|ing)?|error|denied|invalid|refused|unable|cannot|abnormal(?:ly)?|panic|segfault)\b",
    re.IGNORECASE,
)

TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S,%f",
    "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%d/%b/%Y:%H:%M:%S %z", "%a %b %d %H:%M:%S %Y", "%a %b %d %H:%M:%S.%f %Y",
    "%b %d %H:%M:%S", "%b %d %H:%M", "%Y/%m/%d %H:%M:%S",
    "%Y%m%d-%H:%M:%S:%f", "%m.%d %H:%M:%S", "%m.%d %H:%M",
]
YEARLESS_FORMATS = {"%b %d %H:%M:%S", "%b %d %H:%M", "%m.%d %H:%M:%S", "%m.%d %H:%M"}
YEARLESS_PROFILES = {"syslog_rfc3164": "%b %d %H:%M:%S", "proxifier_bracketed": "%m.%d %H:%M:%S"}


def normalize_level(raw: Optional[str]) -> str:
    if not raw:
        return "UNSPECIFIED"
    level = raw.strip().upper()
    level = LEVEL_ALIASES.get(level, level)
    return level if level in KNOWN_LEVELS else "UNSPECIFIED"


def parse_timestamp(text: str, default_year: int = 2000) -> Optional[float]:
    """Return Unix epoch seconds (UTC) or None. Yearless formats get default_year."""
    if not text:
        return None
    candidate = re.sub(r"\s+", " ", text.strip())
    if re.fullmatch(r"\d{10}(?:\.\d+)?", candidate):
        return float(candidate)
    # Python's %z does not accept a trailing Z.
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+0000"
    for fmt in TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(candidate, fmt)
        except ValueError:
            continue
        if fmt in YEARLESS_FORMATS:
            parsed = parsed.replace(year=default_year)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


@dataclass
class FormatProfile:
    name: str
    pattern: re.Pattern
    extract: Callable[[re.Match], dict]


def _python_logging(m: re.Match) -> dict:
    return {"ts": m["ts"], "logger": m["logger"], "level": m["level"], "message": m["msg"]}


def _log4j_bracket(m: re.Match) -> dict:
    logger = m["logger"]
    message = m["msg_b"] if m["msg_b"] is not None else m["msg_a"]
    if not logger:
        # Zookeeper style: [QuorumPeer[myid=1]/0:0:...:2181:FastLeaderElection@774]
        thread = m["thread"] or ""
        last = thread.split(":")[-1]
        logger = re.sub(r"@\d+$", "", last) or thread
    return {"ts": m["ts"], "logger": logger, "level": m["level"], "message": message or ""}


def _iso_severity(m: re.Match) -> dict:
    return {"ts": m["ts"], "logger": m["logger"] or "root", "level": m["level"], "message": m["msg"]}


def _syslog(m: re.Match) -> dict:
    message = m["msg"]
    level = "UNSPECIFIED"
    return {"ts": m["ts"], "logger": m["logger"], "level": level, "message": message, "host": m["host"]}


def _access_combined(m: re.Match) -> dict:
    status = int(m["status"])
    level = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
    request = m["request"] or ""
    parts = request.split()
    path = parts[1] if len(parts) > 1 else "/"
    segment = path.split("?")[0].strip("/").split("/")[0] or "root"
    return {"ts": m["ts"], "logger": f"http:/{segment}", "level": level,
            "message": f'{request} status={status} size={m["size"]} client={m["host"]}'}


def _openstack(m: re.Match) -> dict:
    return {"ts": m["ts"], "logger": m["logger"], "level": m["level"], "message": m["msg"]}


def _healthapp(m: re.Match) -> dict:
    day, clock = m["ts"].split("-", 1)
    hour, minute, second, millis = (int(x) for x in clock.split(":"))
    return {"ts": f"{day}-{hour:02d}:{minute:02d}:{second:02d}:{millis:03d}", "logger": m["logger"], "level": "UNSPECIFIED", "message": m["msg"]}


def _hpc(m: re.Match) -> dict:
    stamp = datetime.fromtimestamp(int(m["epoch"]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return {"ts": stamp, "logger": m["source"], "level": "UNSPECIFIED", "message": f'{m["fields"]} | {m["msg"]}'}


def _proxifier(m: re.Match) -> dict:
    return {"ts": m["ts"], "logger": m["logger"], "level": "UNSPECIFIED", "message": m["msg"]}


EMBEDDED_LEVEL_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[.,]\d+\s+(?P<lv>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\b"),
    re.compile(r'"s":"(?P<mongo>[FEWID])\d?"'),
    re.compile(r'\blevel=(?P<lv>\w+)', re.IGNORECASE),
    re.compile(r'"(?:severity|level)":\s*"(?P<lv>\w+)"', re.IGNORECASE),
    re.compile(r"^\[?(?P<lv>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\]?[\s:]"),
]
MONGO_SEVERITY = {"F": "CRITICAL", "E": "ERROR", "W": "WARN", "I": "INFO", "D": "DEBUG"}


def _embedded_level(message: str) -> str:
    """Severity written inside the message by the application's own logger (Spring, Go kit, MongoDB JSON)."""
    for pattern in EMBEDDED_LEVEL_PATTERNS:
        m = pattern.search(message)
        if m:
            if m.groupdict().get("mongo"):
                return MONGO_SEVERITY[m["mongo"]]
            return m["lv"]
    return "UNSPECIFIED"


def _container_aggregate(m: re.Match) -> dict:
    return {"ts": m["ts"], "logger": m["logger"], "level": _embedded_level(m["msg"]), "message": m["msg"]}


def _bracketed(m: re.Match) -> dict:
    return {"ts": m["ts"], "logger": m["logger"] or "root", "level": m["level"], "message": m["msg"]}


LEVEL_TOKEN = r"(?:TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|ERR|FATAL|CRITICAL|CRIT|EMERG|ALERT)"

FORMAT_PROFILES = [
    FormatProfile(
        "python_logging",
        re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+-\s+(?P<logger>[\w.\-/\[\]]+)\s+-\s+"
                   r"(?P<level>[A-Z]+)\s+-\s+(?P<msg>.*)$"),
        _python_logging,
    ),
    FormatProfile(
        "log4j_bracket",
        re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.]\d{3})\s+(?:-\s+)?(?P<level>[A-Z]+)\s+"
                   r"\[(?P<thread>.*?)\]\s+(?:-\s+(?P<msg_a>.*)|(?P<logger>[\w.$]+):\s*(?P<msg_b>.*))$"),
        _log4j_bracket,
    ),
    FormatProfile(
        "iso8601_with_severity",
        re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+\[?"
                   rf"(?P<level>{LEVEL_TOKEN})\]?\s+(?:(?P<logger>[\w.\-]+):\s+)?(?P<msg>.*)$", re.IGNORECASE),
        _iso_severity,
    ),
    FormatProfile(
        "syslog_rfc3164",
        re.compile(r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+"
                   r"(?P<logger>[^\s:\[]+?(?:\([^)]*\))?(?: [\d.]+)?)(?:\[(?P<pid>\d+)\])?:\s*(?P<msg>.*)$"),
        _syslog,
    ),
    FormatProfile(
        "access_log_combined",
        re.compile(r'^(?P<host>\S+) \S+ \S+ \[(?P<ts>\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4})\] '
                   r'"(?P<request>[^"]*)" (?P<status>\d{3}) (?P<size>\S+)(?: "(?P<referer>[^"]*)" "(?P<agent>[^"]*)")?'),
        _access_combined,
    ),
    FormatProfile(
        "openstack_service_log",
        re.compile(r"^(?P<source>\S+\.log\S*) (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) (?P<pid>\d+) "
                   rf"(?P<level>{LEVEL_TOKEN}) (?P<logger>\S+) (?:\[[^\]]*\] )?(?P<msg>.*)$"),
        _openstack,
    ),
    FormatProfile(
        "pipe_delimited_component",
        re.compile(r"^(?P<ts>\d{8}-\d{1,2}:\d{1,2}:\d{1,2}:\d{1,3})\|(?P<logger>[^|]+)\|(?P<pid>\d+)\|(?P<msg>.*)$"),
        _healthapp,
    ),
    FormatProfile(
        "cluster_node_epoch",
        re.compile(r"^(?P<logid>\d+) (?P<source>\S+) (?P<fields>.+?) (?P<epoch>\d{10}) (?P<flag>\d) (?P<msg>.*)$"),
        _hpc,
    ),
    FormatProfile(
        "proxifier_bracketed",
        re.compile(r"^\[(?P<ts>\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\] (?P<logger>.+?) - (?P<msg>.*)$"),
        _proxifier,
    ),
    FormatProfile(
        "container_log_aggregate",
        re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3,9}Z) (?P<logger>[a-z0-9][a-z0-9\-_.]*) (?P<msg>.*)$"),
        _container_aggregate,
    ),
    FormatProfile(
        "bracketed_timestamp",
        re.compile(rf"^\[(?P<ts>[^\]]+)\]\s+(?:\[?(?P<level>{LEVEL_TOKEN})\]?\s+)?(?:\[(?P<logger>[^\]]+)\]\s+)?"
                   r"(?P<msg>.*)$", re.IGNORECASE),
        _bracketed,
    ),
]


@dataclass
class ParsedLog:
    format_name: str
    match_rate: float
    events: list = field(default_factory=list)
    detection_scores: dict = field(default_factory=dict)


def stratified_sample(lines: list[str], size: int = SAMPLE_LINES) -> list[str]:
    non_empty = [ln for ln in lines if ln.strip()]
    if len(non_empty) <= size:
        return non_empty
    third = size // 3
    middle = len(non_empty) // 2
    return non_empty[:third] + non_empty[middle - third // 2: middle + (third - third // 2)] + non_empty[-(size - 2 * third):]


def detect_format(lines: list[str]) -> tuple[Optional[FormatProfile], dict]:
    sample = stratified_sample(lines)
    scores = {}
    for profile in FORMAT_PROFILES:
        hits = sum(1 for ln in sample if profile.pattern.match(ln))
        scores[profile.name] = hits / max(1, len(sample))
    best = None
    best_rate = MATCH_RATE_THRESHOLD
    for profile in FORMAT_PROFILES:  # priority order breaks ties
        if scores[profile.name] > best_rate:
            best, best_rate = profile, scores[profile.name]
    return best, scores


def _infer_yearless_epochs(events: list[dict], base_year: int, fmt: str = "%b %d %H:%M:%S") -> None:
    """Assign years to yearless timestamps, rolling over when the month decreases."""
    year = base_year
    previous_month = None
    for ev in events:
        if ev["level"] == "RAW" and ev.get("parent") is not None:
            continue
        try:
            parsed = datetime.strptime(re.sub(r"\s+", " ", ev["ts_str"]), fmt)
        except ValueError:
            continue
        if previous_month is not None and parsed.month < previous_month:
            year += 1
        previous_month = parsed.month
        ev["epoch"] = parsed.replace(year=year, tzinfo=timezone.utc).timestamp()


def parse_log_lines(lines: list[str], base_year_for_yearless: int = 2005) -> ParsedLog:
    profile, scores = detect_format(lines)
    events: list[dict] = []
    if profile is None:
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            events.append({"idx": len(events), "line_no": line_no, "ts_str": "", "epoch": None, "logger": "unknown",
                           "level": "RAW", "message": line.rstrip("\n"), "raw": line.rstrip("\n"), "parent": None,
                           "error_like": bool(ERROR_LIKE_PATTERN.search(line))})
        return ParsedLog("raw_fallback", 0.0, events, scores)

    parent_index = None
    for line_no, line in enumerate(lines, start=1):
        raw = line.rstrip("\n")
        if not raw.strip():
            continue
        m = profile.pattern.match(raw)
        if m:
            fields = profile.extract(m)
            level = normalize_level(fields.get("level"))
            message = (fields.get("message") or "").strip()
            event = {
                "idx": len(events), "line_no": line_no, "ts_str": fields["ts"].strip(),
                "epoch": parse_timestamp(fields["ts"], default_year=base_year_for_yearless),
                "logger": (fields.get("logger") or "unknown").strip(), "level": level, "message": message,
                "raw": raw, "parent": None,
            }
            event["error_like"] = level in ("ERROR", "CRITICAL") or (
                level == "UNSPECIFIED" and bool(ERROR_LIKE_PATTERN.search(message)))
            parent_index = event["idx"]
            events.append(event)
        else:
            parent = events[parent_index] if parent_index is not None else None
            events.append({
                "idx": len(events), "line_no": line_no, "ts_str": parent["ts_str"] if parent else "",
                "epoch": parent["epoch"] if parent else None, "logger": parent["logger"] if parent else "unknown",
                "level": "RAW", "message": raw.strip(), "raw": raw, "parent": parent_index, "error_like": False,
            })
    if profile.name in YEARLESS_PROFILES:
        _infer_yearless_epochs(events, base_year_for_yearless, YEARLESS_PROFILES[profile.name])
        for ev in events:
            if ev["level"] == "RAW" and ev["parent"] is not None:
                ev["epoch"] = events[ev["parent"]]["epoch"]
    return ParsedLog(profile.name, scores[profile.name], events, scores)


def parse_log_file(path: str, base_year_for_yearless: int = 2005) -> ParsedLog:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return parse_log_lines(lines, base_year_for_yearless)


def format_event(event: dict, max_chars: int = 300) -> str:
    text = f'{event["ts_str"]} {event["level"]} {event["logger"]}: {event["message"]}'
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
