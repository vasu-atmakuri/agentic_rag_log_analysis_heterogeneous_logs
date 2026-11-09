"""ReAct-style tool-using agent over a prepared log workspace.

Four tools: search_logs, get_stats, get_timeline, get_samples. At most 10 model turns per
question; if no final answer is produced, one last call without tools forces an answer.
Older tool observations are elided when the conversation would exceed the context budget.
"""
from __future__ import annotations

import json
import re
import time

from .local_language_model_backend import ContextLengthExceeded, estimate_tokens
from .log_format_detection_and_parsing import format_event
from .log_source_characterization import final_system_prompt
from .multi_type_chunk_builder import CHUNK_TYPES
from .question_answering_methods import UsageTally, ensure_final_answer, extract_final_answer

MAX_ITERATIONS = 10
SEARCH_TOP_K = 5
SEARCH_CHUNK_CHARS = 600
MAX_TOOL_OUTPUT_CHARS = 2400
MAX_PAGE_SIZE = 40

TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "search_logs",
        "description": "Semantic search over indexed log chunks. Returns the top 5 matching excerpts with time ranges.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Natural language search query"},
            "chunk_type": {"type": "string", "enum": ["any", *CHUNK_TYPES],
                           "description": "Restrict to one chunk type, or 'any'"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_stats",
        "description": "Precomputed exact statistics of the whole log: overview, level counts, per-logger counts and error rates, hourly counts, error bursts, frequent message templates, key entities.",
        "parameters": {"type": "object", "properties": {
            "section": {"type": "string", "enum": ["overview", "loggers", "hourly", "bursts", "templates", "entities"]}},
            "required": ["section"]}}},
    {"type": "function", "function": {
        "name": "get_timeline",
        "description": "Exact filtered listing of log events in time order. Returns the total number of matching events and one page of them. Times use the same format as the log timestamps.",
        "parameters": {"type": "object", "properties": {
            "level": {"type": "string", "description": "DEBUG, INFO, WARN, ERROR, CRITICAL, 'error_like' (error or failure messages), or omit for all levels"},
            "keyword": {"type": "string", "description": "Case-insensitive substring the message must contain"},
            "logger": {"type": "string", "description": "Exact logger/component name"},
            "start_time": {"type": "string"}, "end_time": {"type": "string"},
            "page": {"type": "integer", "description": "Page number starting at 1"},
            "page_size": {"type": "integer", "description": "Events per page, at most 40"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "get_samples",
        "description": "Representative events for a logger and/or message pattern, spread across the log.",
        "parameters": {"type": "object", "properties": {
            "logger": {"type": "string"}, "pattern": {"type": "string", "description": "Case-insensitive substring"},
            "kind": {"type": "string", "enum": ["error", "warn", "info", "any"]},
            "n": {"type": "integer", "description": "Number of samples, at most 10"}},
            "required": []}}},
]


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 40] + "\n...[output truncated]"


class LogToolExecutor:
    def __init__(self, workspace, excluded_chunk_types=frozenset(), disabled_tools=frozenset()):
        self.ws = workspace
        self.excluded = frozenset(excluded_chunk_types)
        self.disabled = frozenset(disabled_tools)
        self.evidence_lines: set[int] = set()

    def run(self, name: str, args: dict) -> str:
        if name in self.disabled:
            return f"Tool '{name}' is not available."
        try:
            handler = getattr(self, f"tool_{name}")
        except AttributeError:
            return f"Unknown tool '{name}'. Available: search_logs, get_stats, get_timeline, get_samples."
        try:
            return _truncate(handler(**{k: v for k, v in (args or {}).items() if v not in (None, "")}))
        except TypeError as exc:
            return f"Invalid arguments for {name}: {exc}"

    def time_range_note(self) -> str:
        s = self.ws.stats
        return (f"The log covers {s['first_timestamp']} to {s['last_timestamp']}; give times in that same format "
                "and within that range.")

    def tool_search_logs(self, query: str = "", chunk_type: str = "any", **_) -> str:
        only = chunk_type if chunk_type in CHUNK_TYPES else None
        if only in self.excluded:
            return f"No chunks of type {only} are available."
        results = self.ws.multi_type_index.search(str(query), SEARCH_TOP_K, excluded_types=self.excluded, only_type=only)
        if not results:
            return "No matching excerpts."
        parts = []
        for chunk, score in results:
            self.evidence_lines.update(chunk["line_numbers"])
            body = chunk["text"] if len(chunk["text"]) <= SEARCH_CHUNK_CHARS else chunk["text"][:SEARCH_CHUNK_CHARS] + " ..."
            parts.append(f"(score {score:.3f})\n{body}")
        return "\n\n".join(parts)

    def tool_get_stats(self, section: str = "overview", **_) -> str:
        s = self.ws.stats
        if section == "loggers":
            rows = [f'{name}: events={v["events"]}, error_like={v["error_like"]}, warn={v["warn"]}, error_rate={v["error_rate"]}'
                    for name, v in s["logger_statistics"].items()]
            return "Per-logger statistics (sorted by event count):\n" + "\n".join(rows[:60])
        if section == "hourly":
            rows = [f"{h}: events={c}, error_like={s['hourly_error_counts'].get(h, 0)}" for h, c in s["hourly_event_counts"].items()]
            return "Hourly counts (UTC hour buckets, same clock as the log):\n" + "\n".join(rows[:80]) + (
                "\n...[more hours exist; use get_timeline with a time range]" if len(rows) > 80 else "")
        if section == "bursts":
            return json.dumps({"mean_error_like_per_active_hour": s["mean_errors_per_active_hour"],
                               "burst_hours": s["error_burst_hours"]})
        if section == "templates":
            return "\n".join(f'[{t["count"]}] {t["template"]}' for t in s["top_message_templates"])
        if section == "entities":
            c = self.ws.characterization
            return json.dumps({"key_entities": c.get("key_entities", []), "key_event_types": c.get("key_event_types", [])})
        return json.dumps({k: s[k] for k in ("total_events", "first_timestamp", "last_timestamp", "time_span_hours",
                                              "level_distribution")}) + f"\nlog format: {self.ws.format_name}"

    def _filter(self, level=None, keyword=None, logger=None, start_time=None, end_time=None):
        start = self.ws.resolve_user_time(start_time) if start_time else None
        end = self.ws.resolve_user_time(end_time) if end_time else None
        level_norm = str(level).upper() if level else None
        if level_norm in ("ANY", "ALL", "*", ""):
            level_norm = None
        if level_norm == "WARNING":
            level_norm = "WARN"
        key = str(keyword).lower() if keyword else None
        out = []
        for e in self.ws.primary:
            if level_norm:
                if level_norm == "ERROR_LIKE":
                    if not e["error_like"]:
                        continue
                elif e["level"] != level_norm:
                    continue
            if logger and e["logger"] != logger:
                continue
            if key and key not in e["message"].lower() and key not in e["raw"].lower():
                continue
            if start is not None and (e["epoch"] is None or e["epoch"] < start):
                continue
            if end is not None and (e["epoch"] is None or e["epoch"] > end):
                continue
            out.append(e)
        out.sort(key=lambda e: (e["epoch"] is None, e["epoch"] or 0.0))
        notes = []
        if start_time and start is None:
            notes.append(f"could not parse start_time '{start_time}'")
        if end_time and end is None:
            notes.append(f"could not parse end_time '{end_time}'")
        return out, notes

    def tool_get_timeline(self, level=None, keyword=None, logger=None, start_time=None, end_time=None,
                          page=1, page_size=20, **_) -> str:
        matches, notes = self._filter(level, keyword, logger, start_time, end_time)
        try:
            page = max(1, int(page))
            page_size = max(1, min(MAX_PAGE_SIZE, int(page_size)))
        except (TypeError, ValueError):
            page, page_size = 1, 20
        window = matches[(page - 1) * page_size: page * page_size]
        self.evidence_lines.update(e["line_no"] for e in window)
        header = f"total_matching_events={len(matches)}; page {page} of {max(1, -(-len(matches) // page_size))}"
        if not matches:
            notes.append(self.time_range_note())
        if notes:
            header += "; NOTE: " + "; ".join(notes)
        return header + "\n" + "\n".join(format_event(e, 260) for e in window)

    def tool_get_samples(self, logger=None, pattern=None, kind="any", n=5, **_) -> str:
        level = {"error": "error_like", "warn": "WARN", "info": "INFO"}.get(str(kind).lower())
        matches, _ = self._filter(level=level, keyword=pattern, logger=logger)
        try:
            n = max(1, min(10, int(n)))
        except (TypeError, ValueError):
            n = 5
        if not matches:
            return "No matching events. " + self.time_range_note()
        step = max(1, len(matches) // n)
        picks = matches[::step][:n]
        self.evidence_lines.update(e["line_no"] for e in picks)
        return f"{len(matches)} matching events; {len(picks)} samples:\n" + "\n".join(format_event(e, 260) for e in picks)


LEVEL_SYNONYMS = {"WARNING": "WARN", "ERR": "ERROR", "FATAL": "CRITICAL", "CRIT": "CRITICAL", "SEVERE": "ERROR",
                  "ERRORS": "ERROR", "WARNINGS": "WARN", "FAILURE": "ERROR_LIKE", "FAILURES": "ERROR_LIKE"}


class RobustLogToolExecutor(LogToolExecutor):
    """Tool-robustness layer: validates and corrects arguments before filtering and explains empty results.

    Component names: exact match, else case-insensitive match, else a unique substring match, else the closest
    name by string similarity (difflib ratio >= 0.6); the correction is reported to the model. An unknown name
    with no close match returns the list of known components. Level synonyms are normalized. A start time at or
    after the end time returns an explanatory error. A filter that matches nothing reports how many events match
    when the time filter or the component filter is dropped.
    """

    def _components(self):
        if not hasattr(self, "_component_names"):
            self._component_names = list(self.ws.stats["logger_statistics"].keys())
        return self._component_names

    def resolve_component(self, name):
        import difflib

        if not name:
            return None, None
        names = self._components()
        name = str(name).strip()
        if name in names:
            return name, None
        lower = {n.lower(): n for n in names}
        if name.lower() in lower:
            return lower[name.lower()], f"component '{name}' matched as '{lower[name.lower()]}'"
        contains = [n for n in names if name.lower() in n.lower()]
        if len(contains) == 1:
            return contains[0], f"component '{name}' matched as '{contains[0]}'"
        close = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
        if close:
            return close[0], f"component '{name}' not found; using closest name '{close[0]}'"
        listed = ", ".join(names[:40])
        return False, f"component '{name}' does not exist. Known components: {listed}"

    def _normalized_level(self, level):
        if not level:
            return None
        lv = str(level).strip().upper()
        return LEVEL_SYNONYMS.get(lv, lv)

    def _robust_filter(self, level=None, keyword=None, logger=None, start_time=None, end_time=None):
        notes = []
        component, note = self.resolve_component(logger)
        if note:
            notes.append(note)
        if component is False:
            return None, notes
        level = self._normalized_level(level)
        if start_time and end_time:
            start = self.ws.resolve_user_time(start_time)
            end = self.ws.resolve_user_time(end_time)
            if start is not None and end is not None and start >= end:
                notes.append(f"start_time '{start_time}' is not before end_time '{end_time}'; give a window with "
                             "the start earlier than the end")
                return None, notes
        matches, more = self._filter(level, keyword, component, start_time, end_time)
        notes.extend(more)
        if not matches:
            hints = []
            if start_time or end_time:
                without_time, _ = self._filter(level, keyword, component, None, None)
                hints.append(f"{len(without_time)} events match without the time filter")
            if component:
                without_component, _ = self._filter(level, keyword, None, start_time, end_time)
                hints.append(f"{len(without_component)} events match without the component filter")
            if hints:
                notes.append("no events matched; " + "; ".join(hints))
        return matches, notes

    def tool_get_timeline(self, level=None, keyword=None, logger=None, start_time=None, end_time=None,
                          page=1, page_size=20, **_) -> str:
        matches, notes = self._robust_filter(level, keyword, logger, start_time, end_time)
        if matches is None:
            return "total_matching_events=0; NOTE: " + "; ".join(notes)
        try:
            page = max(1, int(page))
            page_size = max(1, min(MAX_PAGE_SIZE, int(page_size)))
        except (TypeError, ValueError):
            page, page_size = 1, 20
        window = matches[(page - 1) * page_size: page * page_size]
        self.evidence_lines.update(e["line_no"] for e in window)
        header = f"total_matching_events={len(matches)}; page {page} of {max(1, -(-len(matches) // page_size))}"
        if not matches:
            notes.append(self.time_range_note())
        if notes:
            header += "; NOTE: " + "; ".join(notes)
        return header + "\n" + "\n".join(format_event(e, 260) for e in window)

    def tool_get_samples(self, logger=None, pattern=None, kind="any", n=5, **_) -> str:
        level = {"error": "error_like", "warn": "WARN", "info": "INFO"}.get(str(kind).lower())
        matches, notes = self._robust_filter(level=level, keyword=pattern, logger=logger)
        prefix = ("NOTE: " + "; ".join(notes) + "\n") if notes else ""
        if not matches:
            return prefix + "No matching events. " + self.time_range_note()
        try:
            n = max(1, min(10, int(n)))
        except (TypeError, ValueError):
            n = 5
        step = max(1, len(matches) // n)
        picks = matches[::step][:n]
        self.evidence_lines.update(e["line_no"] for e in picks)
        return prefix + f"{len(matches)} matching events; {len(picks)} samples:\n" + "\n".join(format_event(e, 260) for e in picks)


def _messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(str(m.get("content", ""))) + 8 for m in messages)


def _fit_to_budget(messages: list[dict], budget: int) -> int:
    elided = 0
    for m in messages:
        if _messages_tokens(messages) <= budget:
            break
        if m["role"] == "tool" and not m["content"].startswith("[earlier tool output elided"):
            m["content"] = "[earlier tool output elided to fit the context window]"
            elided += 1
    return elided


def _elide_all_but_last_observation(messages: list[dict]) -> int:
    tool_positions = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    count = 0
    for i in tool_positions[:-1]:
        if not messages[i]["content"].startswith("[earlier tool output elided"):
            messages[i]["content"] = "[earlier tool output elided to fit the context window]"
            count += 1
    if tool_positions:
        last = messages[tool_positions[-1]]
        last["content"] = last["content"][:1200]
    return count


def answer_tool_using_agent(workspace, backend, question: str, thinking: bool = False,
                            excluded_chunk_types=frozenset(), disabled_tools=frozenset(),
                            robust_tools: bool = False) -> dict:
    started = time.perf_counter()
    executor_class = RobustLogToolExecutor if robust_tools else LogToolExecutor
    executor = executor_class(workspace, excluded_chunk_types, disabled_tools)
    tool_definitions = [t for t in TOOL_DEFINITIONS if t["function"]["name"] not in set(disabled_tools)]
    s = workspace.stats
    time_context = (f"\n\nThe log covers {s['first_timestamp']} to {s['last_timestamp']}. When a tool takes a time, "
                    "write it in the same format as these log timestamps. If a tool returns no matching events, check "
                    "your arguments (time range, level, component name) and try again before answering.")
    messages = [{"role": "system", "content": final_system_prompt(workspace.characterization, "agent") + time_context},
                {"role": "user", "content": f"QUESTION: {question}"}]
    budget = backend.prompt_budget_tokens(thinking) - 900  # tool schemas also occupy context
    tally = UsageTally()
    trace, tool_calls, elided = [], 0, 0
    final_text, finished, last_response = "", False, None

    def call(msgs, with_tools: bool):
        nonlocal elided
        for attempt in range(3):
            try:
                return backend.chat(msgs, tools=tool_definitions if with_tools else None, thinking=thinking)
            except ContextLengthExceeded:
                tally.context_retries += 1
                elided += _elide_all_but_last_observation(msgs)
                if attempt == 1:
                    for m in msgs:
                        if m["role"] == "tool":
                            m["content"] = m["content"][:400]
        raise ContextLengthExceeded("agent conversation does not fit the context window")

    for iteration in range(MAX_ITERATIONS):
        elided += _fit_to_budget(messages, budget)
        response = call(messages, True)
        tally.add(response)
        last_response = response
        if not response.tool_calls:
            final_text, finished = response.content, True
            trace.append({"iteration": iteration, "type": "final", "content": response.content[:2000]})
            break
        messages.append({"role": "assistant", "content": response.content or "",
                         "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in response.tool_calls]})
        for name, args in response.tool_calls:
            observation = executor.run(name, args)
            tool_calls += 1
            trace.append({"iteration": iteration, "type": "tool", "tool": name, "arguments": args,
                          "observation_chars": len(observation), "observation_head": observation[:400]})
            messages.append({"role": "tool", "content": observation, "tool_name": name})
    if not finished:
        messages.append({"role": "user", "content": "Tool budget exhausted. Give your final answer now from the evidence gathered."})
        elided += _fit_to_budget(messages, budget + 900)
        last_response = call(messages, False)
        tally.add(last_response)
        final_text = last_response.content
        trace.append({"iteration": MAX_ITERATIONS, "type": "forced_final", "content": final_text[:2000]})
    final_text, extracted = ensure_final_answer(backend, messages, last_response, tally, thinking)
    return {
        "method": "robust_tool_using_agent" if robust_tools else "tool_using_agent", "response_text": final_text, "final_answer": extract_final_answer(final_text),
        "answer_extraction_followup": extracted, "thinking_chars": tally.thinking_chars, "llm_calls": tally.llm_calls,
        "tool_calls": tool_calls, "prompt_tokens": tally.prompt_tokens, "completion_tokens": tally.completion_tokens,
        "llm_seconds": round(tally.llm_seconds, 3), "wall_seconds": round(time.perf_counter() - started, 3),
        "context_overflow_risk": tally.overflow, "output_truncated": tally.truncated,
        "context_retries": tally.context_retries, "elided_observations": elided, "hit_iteration_limit": not finished,
        "evidence_line_numbers": sorted(executor.evidence_lines),
        "tools_used": [t["tool"] for t in trace if t["type"] == "tool"], "trace": trace,
    }
