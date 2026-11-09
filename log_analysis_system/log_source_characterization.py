"""Characterization of an unseen log source.

Three conditions are supported so the contribution of automatic characterization can be
measured directly:
  automatic : the language model reads a statistical summary plus a representative excerpt
              and returns log type, entities, important/noisy loggers and two system prompts.
  generic   : fixed, source-independent prompts; no logger is marked noisy.
  expert    : prompts and logger lists written by hand by the author for that source.
In every condition the same answer-format instruction is appended, so grading is identical.
"""
from __future__ import annotations

import json
import os
import re

import yaml

from .statistical_log_profiler import compact_statistics_summary, primary_events
from .log_format_detection_and_parsing import format_event

REPRESENTATIVE_SAMPLE_LINES = 80
MAX_GENERATED_PROMPT_CHARS = 1500

ANSWER_FORMAT_INSTRUCTION = (
    "Base every statement only on log evidence you were given. If the evidence is insufficient, say so. "
    "Finish your response with exactly one line of the form 'FINAL ANSWER: <answer>', where <answer> is short "
    "(a number, a component name, a timestamp, or a brief phrase)."
)

GENERIC_RAG_PROMPT = (
    "You are a log analysis assistant. You are given excerpts retrieved from a log file. "
    "Answer the user's question using only those excerpts."
)
GENERIC_AGENT_PROMPT = (
    "You are a log analysis assistant with tools to search and inspect a log file. "
    "Use the tools to gather evidence before answering the user's question."
)

CHARACTERIZATION_INSTRUCTION = """You are given a statistical summary and a representative excerpt of a log file from an unknown system.
Return a JSON object with exactly these fields:
  "log_type": short descriptive label of the log source,
  "system_description": one paragraph describing the system and its components,
  "key_entities": list of important identifiers (hosts, versions, endpoints, components),
  "key_event_types": list of notable event patterns,
  "important_loggers": list of logger/component names (copied exactly from the summary) that carry high-signal events,
  "noisy_loggers": list of logger/component names (copied exactly) that mostly emit routine low-signal events,
  "rag_system_prompt": a system prompt (under 1200 characters) for answering questions from retrieved excerpts of this specific log,
  "agent_system_prompt": a system prompt (under 1200 characters) for a tool-using analyst investigating this specific log; mention which components matter and typical failure modes.
Return only the JSON object."""


def representative_sample(events: list[dict], n: int = REPRESENTATIVE_SAMPLE_LINES) -> list[str]:
    primary = primary_events(events)
    if not primary:
        return []
    errors = [e for e in primary if e["error_like"]]
    picks = []
    step = max(1, len(primary) // max(1, n - min(len(errors), n // 4)))
    picks.extend(primary[::step])
    picks.extend(errors[:: max(1, len(errors) // max(1, n // 4))] if errors else [])
    seen, ordered = set(), []
    for e in sorted(picks, key=lambda x: x["idx"]):
        if e["idx"] not in seen:
            seen.add(e["idx"])
            ordered.append(format_event(e, 160))
    return ordered[:n]


def _clean_list(value, known: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if str(v) in known]


def _bounded(text, fallback: str) -> str:
    text = str(text or "").strip()
    return text[:MAX_GENERATED_PROMPT_CHARS] if len(text) >= 40 else fallback


CHARACTERIZATION_MAX_OUTPUT_TOKENS = 2048
CHARACTERIZATION_VERSION = 2
# Version 2: the characterization call has its own 2,048-token output budget (version 1 shared the 768-token
# answer budget, which truncated the JSON for several models so the generic fallback was used silently), and
# one retry is made if the JSON still cannot be parsed. Attempts and the parse outcome are recorded.


def _parse_json_object(text: str) -> dict:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}


def automatic_characterization(backend, events: list[dict], stats: dict) -> dict:
    summary = compact_statistics_summary(stats)
    sample = "\n".join(representative_sample(events))
    user = f"STATISTICAL SUMMARY\n{summary}\n\nREPRESENTATIVE EXCERPT\n{sample}"
    messages = [{"role": "system", "content": CHARACTERIZATION_INSTRUCTION}, {"role": "user", "content": user}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "duration_seconds": 0.0, "attempts": 0, "truncated": []}
    payload = {}
    for attempt in range(2):
        if attempt == 1:
            messages = messages + [{"role": "user", "content": "Your previous output was not a complete JSON object. "
                                    "Return the JSON object again, complete, with each system prompt under 800 characters."}]
        result = backend.chat(messages, thinking=False, json_format=True,
                              max_output_tokens=CHARACTERIZATION_MAX_OUTPUT_TOKENS)
        usage["attempts"] += 1
        usage["prompt_tokens"] += result.prompt_tokens
        usage["completion_tokens"] += result.completion_tokens
        usage["duration_seconds"] = round(usage["duration_seconds"] + result.duration_seconds, 3)
        usage["truncated"].append(bool(getattr(result, "output_truncated", False)))
        payload = _parse_json_object(result.content)
        if payload:
            break
    parse_ok = bool(payload)
    known = set(stats["logger_statistics"].keys())
    return {
        "condition": "automatic",
        "parse_ok": parse_ok,
        "log_type": str(payload.get("log_type", "unknown"))[:120],
        "system_description": str(payload.get("system_description", ""))[:1500],
        "key_entities": [str(x)[:120] for x in payload.get("key_entities", [])][:30] if isinstance(payload.get("key_entities"), list) else [],
        "key_event_types": [str(x)[:160] for x in payload.get("key_event_types", [])][:20] if isinstance(payload.get("key_event_types"), list) else [],
        "important_loggers": _clean_list(payload.get("important_loggers"), known),
        "noisy_loggers": _clean_list(payload.get("noisy_loggers"), known),
        "rag_system_prompt": _bounded(payload.get("rag_system_prompt"), GENERIC_RAG_PROMPT),
        "agent_system_prompt": _bounded(payload.get("agent_system_prompt"), GENERIC_AGENT_PROMPT),
        "characterization_version": CHARACTERIZATION_VERSION,
        "usage": usage,
    }


def generic_characterization(stats: dict, top_loggers: int = 7) -> dict:
    loggers = list(stats["logger_statistics"].keys())[:top_loggers]
    return {"condition": "generic", "parse_ok": True, "log_type": "unknown", "system_description": "",
            "key_entities": [], "key_event_types": [], "important_loggers": loggers, "noisy_loggers": [],
            "rag_system_prompt": GENERIC_RAG_PROMPT, "agent_system_prompt": GENERIC_AGENT_PROMPT, "usage": {}}


def expert_characterization(config_path: str, stats: dict) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Expert configuration not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    text = json.dumps(cfg)
    if "TODO" in text:
        raise ValueError(f"Expert configuration {config_path} still contains TODO placeholders; write it before running.")
    known = set(stats["logger_statistics"].keys())
    return {"condition": "expert", "parse_ok": True, "log_type": cfg.get("log_type", ""),
            "system_description": cfg.get("system_description", ""), "key_entities": cfg.get("key_entities", []),
            "key_event_types": cfg.get("key_event_types", []),
            "important_loggers": _clean_list(cfg.get("important_loggers"), known),
            "noisy_loggers": _clean_list(cfg.get("noisy_loggers"), known),
            "rag_system_prompt": cfg["rag_system_prompt"], "agent_system_prompt": cfg["agent_system_prompt"], "usage": {}}


def effective_important_loggers(characterization: dict, stats: dict, fallback_count: int = 7) -> list[str]:
    loggers = characterization.get("important_loggers") or []
    return loggers if loggers else list(stats["logger_statistics"].keys())[:fallback_count]


def final_system_prompt(characterization: dict, mode: str) -> str:
    base = characterization["agent_system_prompt"] if mode == "agent" else characterization["rag_system_prompt"]
    return f"{base}\n\n{ANSWER_FORMAT_INSTRUCTION}"
