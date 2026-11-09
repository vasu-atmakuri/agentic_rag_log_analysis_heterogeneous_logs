"""Independent reference readers used only to compute gold answers.

They are deliberately separate from the system's parser (simple dataset-specific regular
expressions), so a parsing defect in the system cannot silently make its answers agree
with the gold standard.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

ZOOKEEPER_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - (\w+)\s+\[(.*?)\] - (.*)$")
LINUX_LINE = re.compile(r"^([A-Z][a-z]{2})\s+(\d{1,2}) (\d{2}:\d{2}:\d{2}) (\S+) ([^\s:\[]+?(?:\([^)]*\))?(?: [\d.]+)?)(?:\[\d+\])?:\s*(.*)$")


def read_fault_injected_testbed(manifest_path: str) -> tuple[list[dict], dict]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    events = [{"line_no": e["line_no"], "epoch": e["epoch"], "ts_str": e["ts_str"], "component": e["component"],
               "level": e["level"], "message": e["message"], "tag": e.get("tag")} for e in manifest["events"]]
    return events, manifest


def read_zookeeper(path: str) -> list[dict]:
    events = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            m = ZOOKEEPER_LINE.match(line.rstrip("\n"))
            if not m:
                continue
            stamp, millis, level, thread, message = m.groups()
            component = re.sub(r"@\d+$", "", thread.split(":")[-1])
            epoch = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() + int(millis) / 1000
            level = {"WARNING": "WARN", "FATAL": "CRITICAL"}.get(level, level)
            events.append({"line_no": line_no, "epoch": epoch, "ts_str": f"{stamp},{millis}", "component": component,
                           "level": level, "message": message.strip(), "tag": None})
    return events


def read_linux_syslog(path: str, base_year: int = 2005) -> list[dict]:
    events, year, previous_month = [], base_year, None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            m = LINUX_LINE.match(line.rstrip("\n"))
            if not m:
                continue
            month, day, clock, _host, component, message = m.groups()
            parsed = datetime.strptime(f"{month} {day} {clock}", "%b %d %H:%M:%S")
            if previous_month is not None and parsed.month < previous_month:
                year += 1
            previous_month = parsed.month
            epoch = parsed.replace(year=year, tzinfo=timezone.utc).timestamp()
            events.append({"line_no": line_no, "epoch": epoch, "ts_str": f"{month} {int(day):2d} {clock}",
                           "component": component, "level": "UNSPECIFIED", "message": message.strip(), "tag": None})
    return events


# ----------------------------------------------------------------------------- Loghub-2.0 sources
# Header fields are read with dataset-specific regular expressions written for this purpose; message and
# event-template identity come from the structured CSV annotated by the Loghub-2.0 authors (joined on LineId).
LOGHUB2_LINE_PATTERNS = {
    "Apache": (r"^\[(?P<ts>\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \d{4})\] \[(?P<level>\w+)\] ", "%a %b %d %H:%M:%S %Y", None),
    "Hadoop": (r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (?P<level>[A-Z]+) \[[^\]]*\] (?P<component>[\w.$]+): ",
               "%Y-%m-%d %H:%M:%S,%f", None),
    "HealthApp": (r"^(?P<ts>\d{8}-\d{1,2}:\d{1,2}:\d{1,2}:\d{1,3})\|(?P<component>[^|]+)\|\d+\|", "healthapp", None),
    "HPC": (r"^\d+ (?P<component>\S+) .+? (?P<ts>\d{10}) \d ", "epoch", None),
    "Linux": (r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2} \d{2}:\d{2}:\d{2}) \S+ (?P<component>[^\s:\[]+?(?:\([^)]*\))?(?: [\d.]+)?)(?:\[\d+\])?:",
              "%b %d %H:%M:%S", 2005),
    "Mac": (r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2} \d{2}:\d{2}:\d{2}) \S+ (?P<component>[^\s:\[]+?(?:\([^)]*\))?(?: [\d.]+)?)(?:\[\d+\])?:",
            "%b %d %H:%M:%S", 2017),
    "OpenSSH": (r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2} \d{2}:\d{2}:\d{2}) \S+ (?P<component>[^\s:\[]+)(?:\[\d+\])?:",
                "%b %d %H:%M:%S", 2015),
    "OpenStack": (r"^\S+ (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \d+ (?P<level>[A-Z]+) (?P<component>\S+) ",
                  "%Y-%m-%d %H:%M:%S.%f", None),
    "Proxifier": (r"^\[(?P<ts>\d{2}\.\d{2} \d{2}:\d{2}:\d{2})\] (?P<component>.+?) - ", "%m.%d %H:%M:%S", 2016),
    "Zookeeper": (r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<level>[A-Z]+)\s+\[(?P<thread>.*?)\] - ",
                  "%Y-%m-%d %H:%M:%S,%f", None),
}
LEVEL_MAP = {"WARNING": "WARN", "NOTICE": "INFO", "FATAL": "CRITICAL", "ERR": "ERROR", "EMERG": "CRITICAL",
             "CRIT": "CRITICAL", "ALERT": "CRITICAL"}


def _parse_ts(text: str, fmt: str, year: int | None) -> tuple[float, int]:
    if fmt == "epoch":
        return float(text), 0
    if fmt == "healthapp":
        day, clock = text.split("-", 1)
        h, m, s, ms = (int(x) for x in clock.split(":"))
        dt = datetime.strptime(day, "%Y%m%d").replace(hour=h, minute=m, second=s, microsecond=ms * 1000, tzinfo=timezone.utc)
        return dt.timestamp(), 0
    parsed = datetime.strptime(re.sub(r"\s+", " ", text), fmt)
    return parsed.replace(tzinfo=timezone.utc).timestamp(), parsed.month


def read_loghub2(name: str, log_path: str, structured_csv: str, max_lines: int | None = None) -> list[dict]:
    import csv

    pattern, fmt, base_year = LOGHUB2_LINE_PATTERNS[name]
    regex = re.compile(pattern)
    templates = {}
    with open(structured_csv, "r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            templates[int(row["LineId"])] = (row["EventId"], row["EventTemplate"], row["Content"])
    events, year, previous_month = [], base_year, None
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if max_lines and line_no > max_lines:
                break
            m = regex.match(line.rstrip("\n"))
            if not m or line_no not in templates:
                continue
            groups = m.groupdict()
            epoch, month = _parse_ts(groups["ts"], fmt, base_year)
            if base_year is not None:
                if previous_month is not None and month < previous_month:
                    year += 1
                previous_month = month
                parsed = datetime.strptime(re.sub(r"\s+", " ", groups["ts"]), fmt).replace(year=year, tzinfo=timezone.utc)
                epoch = parsed.timestamp()
            component = groups.get("component")
            if name == "Zookeeper":
                component = re.sub(r"@\d+$", "", groups["thread"].split(":")[-1])
            level = groups.get("level")
            level = LEVEL_MAP.get(level.upper(), level.upper()) if level else "UNSPECIFIED"
            event_id, template, content = templates[line_no]
            events.append({"line_no": line_no, "epoch": epoch, "ts_str": groups["ts"], "component": component or "",
                           "level": level, "message": content.strip(), "event_id": event_id, "event_template": template,
                           "tag": None})
    return events
