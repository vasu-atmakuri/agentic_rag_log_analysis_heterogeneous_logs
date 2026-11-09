"""Local language model backend (Ollama) with token, latency and context accounting.

Every call is deterministic as far as the runtime allows (temperature 0, fixed seed)
and returns usage counters so that cost-free efficiency metrics can be reported.
A scripted backend is provided for offline unit tests of the pipeline logic.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

CHARS_PER_TOKEN_ESTIMATE = 1.8  # measured about 2.1 characters per token for log text on Qwen3; kept conservative


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1


REQUEST_TIMEOUT_SECONDS = 600


@dataclass
class ModelRuntimeSettings:
    model: str = "qwen3:4b"
    num_ctx: int = 8192
    temperature: float = 0.0
    seed: int = 42
    keep_alive: str = "2m"
    max_output_tokens: int = 768
    max_output_tokens_thinking: int = 2048
    host: Optional[str] = None


class ContextLengthExceeded(RuntimeError):
    """The prompt does not fit the context window; callers shrink their evidence and retry."""


@dataclass
class ChatResult:
    content: str
    tool_calls: list = field(default_factory=list)  # list of (name, arguments_dict)
    thinking: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    context_overflow_risk: bool = False
    output_truncated: bool = False


class OllamaChatBackend:
    def __init__(self, settings: ModelRuntimeSettings):
        import ollama  # imported lazily so the package imports without the runtime

        self.settings = settings
        # A single model call never needs 10 minutes when the server is healthy; a hung server raises instead.
        self.client = (ollama.Client(host=settings.host, timeout=REQUEST_TIMEOUT_SECONDS) if settings.host
                       else ollama.Client(timeout=REQUEST_TIMEOUT_SECONDS))

    def prompt_budget_tokens(self, thinking: bool) -> int:
        reserve = self.settings.max_output_tokens_thinking if thinking else self.settings.max_output_tokens
        return self.settings.num_ctx - reserve - 256

    def chat(self, messages: list[dict], tools: Optional[list] = None, thinking: bool = False,
             json_format: bool = False, max_output_tokens: Optional[int] = None) -> ChatResult:
        num_predict = self.settings.max_output_tokens_thinking if thinking else self.settings.max_output_tokens
        if max_output_tokens:
            num_predict = max_output_tokens
        options = {"temperature": self.settings.temperature, "seed": self.settings.seed,
                   "num_ctx": self.settings.num_ctx, "num_predict": num_predict}
        kwargs = {"model": self.settings.model, "messages": messages, "options": options,
                  "keep_alive": self.settings.keep_alive}
        if thinking:
            kwargs["think"] = True
        if tools:
            kwargs["tools"] = tools
        if json_format:
            kwargs["format"] = "json"
        start = time.perf_counter()
        try:
            response = self.client.chat(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if "context" in str(exc).lower() and "exceed" in str(exc).lower():
                raise ContextLengthExceeded(str(exc)) from exc
            raise
        elapsed = time.perf_counter() - start
        message = response.message
        calls = []
        for call in (message.tool_calls or []):
            arguments = call.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append((call.function.name, dict(arguments or {})))
        prompt_tokens = int(response.prompt_eval_count or 0)
        return ChatResult(
            content=message.content or "", tool_calls=calls, thinking=getattr(message, "thinking", "") or "",
            prompt_tokens=prompt_tokens, completion_tokens=int(response.eval_count or 0),
            duration_seconds=elapsed,
            context_overflow_risk=prompt_tokens >= self.settings.num_ctx - num_predict - 16,
            output_truncated=getattr(response, "done_reason", "") == "length",
        )


class ScriptedChatBackend:
    """Deterministic stand-in used only by offline tests; never used for reported results."""

    def __init__(self, settings: ModelRuntimeSettings, script=None):
        self.settings = settings
        self.script = script

    def prompt_budget_tokens(self, thinking: bool) -> int:
        return self.settings.num_ctx - self.settings.max_output_tokens - 256

    def chat(self, messages, tools=None, thinking=False, json_format=False, max_output_tokens=None) -> ChatResult:
        if self.script is not None:
            return self.script(messages, tools, thinking, json_format)
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        if json_format:
            return ChatResult(content=json.dumps({"log_type": "test", "system_description": "test",
                                                 "key_entities": [], "key_event_types": [],
                                                 "important_loggers": [], "noisy_loggers": [],
                                                 "agent_system_prompt": "Answer from tool evidence.",
                                                 "rag_system_prompt": "Answer from the excerpts."}),
                              prompt_tokens=estimate_tokens(prompt), completion_tokens=50)
        return ChatResult(content="FINAL ANSWER: unknown", prompt_tokens=estimate_tokens(prompt), completion_tokens=5)


def build_backend(settings: ModelRuntimeSettings, scripted: bool = False):
    return ScriptedChatBackend(settings) if scripted else OllamaChatBackend(settings)
