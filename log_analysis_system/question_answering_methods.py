"""Answer-generation methods compared in the evaluation.

bm25_line_retrieval        : BM25 over raw log lines, top-5 lines as context (keyword baseline).
long_context_uniform       : as many log lines as fit in the context window, sampled with a
                             uniform stride across the whole file, in original order.
fixed_size_chunk_rag       : dense retrieval over fixed-size 1,024-character chunks (framework-default style).
multi_type_chunk_rag       : dense retrieval over the four-type chunk index, top-8 chunks,
                             characterization-derived RAG system prompt.
tool_using_agent           : multi-turn tool-calling loop (see agentic_tool_use_loop).

Shared rules for every method:
  * same model, decoding settings and answer-format instruction;
  * evidence is packed to an estimated token budget; if the runtime still reports that the prompt
    exceeds the context window, the evidence is shrunk by 20% and the call is retried (up to 4 times);
  * if a response contains no 'FINAL ANSWER:' line, one short follow-up turn asks the model to state
    its final answer from its own response (answer extraction); this is recorded per run.
"""
from __future__ import annotations

import re
import time

from .local_language_model_backend import ContextLengthExceeded, estimate_tokens
from .log_source_characterization import ANSWER_FORMAT_INSTRUCTION, GENERIC_RAG_PROMPT, final_system_prompt
from .log_analysis_workspace import tokenize_for_bm25

BM25_TOP_LINES = 5
RAG_TOP_CHUNKS = 8
SHRINK_STEPS = (1.0, 0.8, 0.64, 0.5)
FINAL_ANSWER_PATTERN = re.compile(r"FINAL ANSWER\s*[:\-]\s*(.+)", re.IGNORECASE)
EXTRACTION_REQUEST = ("State your final answer now as exactly one line in the form 'FINAL ANSWER: <answer>', "
                      "based only on your response above.")


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return re.sub(r"<think>.*", "", text, flags=re.DOTALL)  # unterminated reasoning block


def has_final_answer(text: str) -> bool:
    return bool(FINAL_ANSWER_PATTERN.search(strip_reasoning(text)))


def extract_final_answer(text: str) -> str:
    text = strip_reasoning(text)
    matches = FINAL_ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].strip().strip("*` ").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


class UsageTally:
    def __init__(self):
        self.prompt_tokens = self.completion_tokens = self.llm_calls = 0
        self.llm_seconds = 0.0
        self.overflow = self.truncated = False
        self.thinking_chars = 0
        self.context_retries = 0

    def add(self, response):
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.llm_calls += 1
        self.llm_seconds += response.duration_seconds
        self.overflow |= response.context_overflow_risk
        self.truncated |= response.output_truncated
        self.thinking_chars += len(response.thinking or "")


def ensure_final_answer(backend, messages: list[dict], response, tally: UsageTally, thinking: bool):
    """If the answer line is missing, ask once for it. Returns (full_text, extraction_used)."""
    text = response.content or ""
    if has_final_answer(text):
        return text, False
    follow = messages + [{"role": "assistant", "content": strip_reasoning(text)[-3000:]},
                         {"role": "user", "content": EXTRACTION_REQUEST}]
    try:
        extra = backend.chat(follow, thinking=False)
    except ContextLengthExceeded:
        follow = [messages[0], {"role": "assistant", "content": strip_reasoning(text)[-1500:]},
                  {"role": "user", "content": EXTRACTION_REQUEST}]
        extra = backend.chat(follow, thinking=False)
    tally.add(extra)
    return text + "\n" + (extra.content or ""), True


def _finish(method, text, extraction_used, tally, started, lines, prompt_chars, extra=None):
    out = {
        "method": method, "response_text": text, "final_answer": extract_final_answer(text),
        "answer_extraction_followup": extraction_used, "thinking_chars": tally.thinking_chars,
        "llm_calls": tally.llm_calls, "tool_calls": 0, "prompt_tokens": tally.prompt_tokens,
        "completion_tokens": tally.completion_tokens, "llm_seconds": round(tally.llm_seconds, 3),
        "wall_seconds": round(time.perf_counter() - started, 3), "context_overflow_risk": tally.overflow,
        "output_truncated": tally.truncated, "context_retries": tally.context_retries, "prompt_chars": prompt_chars,
        "evidence_line_numbers": sorted(set(lines)), "trace": [],
    }
    if extra:
        out.update(extra)
    return out


def _single_turn(method, backend, system, question, build_context, thinking, started):
    """build_context(budget_tokens) -> (context_label, context_text, evidence_lines, extra_fields)."""
    tally = UsageTally()
    base_budget = backend.prompt_budget_tokens(thinking) - estimate_tokens(system) - estimate_tokens(question) - 64
    last_error = None
    for scale in SHRINK_STEPS:
        label, context, lines, extra = build_context(int(base_budget * scale))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"{label}\n{context}\n\nQUESTION: {question}\n\n{ANSWER_FORMAT_INSTRUCTION}"}]
        try:
            response = backend.chat(messages, thinking=thinking)
        except ContextLengthExceeded as exc:
            tally.context_retries += 1
            last_error = exc
            continue
        tally.add(response)
        text, extracted = ensure_final_answer(backend, messages, response, tally, thinking)
        return _finish(method, text, extracted, tally, started, lines, sum(len(m["content"]) for m in messages), extra)
    raise last_error


def answer_bm25_line_retrieval(workspace, backend, question: str, thinking: bool = False,
                               top_lines: int = BM25_TOP_LINES) -> dict:
    started = time.perf_counter()
    scores = workspace.bm25.get_scores(tokenize_for_bm25(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_lines]

    def build(budget):
        # keep the highest-ranked lines that fit the prompt budget, then show them in log order
        kept, used = [], 0
        for i in ranked:
            cost = estimate_tokens(workspace.bm25_line_events[i]["raw"][:400]) + 1
            if used + cost > budget:
                break
            kept.append(i)
            used += cost
        events = [workspace.bm25_line_events[i] for i in sorted(kept)]
        return ("LOG LINES RETRIEVED BY KEYWORD SEARCH", "\n".join(e["raw"][:400] for e in events),
                [e["line_no"] for e in events], None)

    return _single_turn("bm25_line_retrieval", backend, f"{GENERIC_RAG_PROMPT}\n\n{ANSWER_FORMAT_INSTRUCTION}",
                        question, build, thinking, started)


def answer_long_context_uniform(workspace, backend, question: str, thinking: bool = False) -> dict:
    started = time.perf_counter()
    events = workspace.events
    lines = [e["raw"][:300] for e in events]
    costs = [estimate_tokens(ln) for ln in lines]
    total = sum(costs)

    def build(budget):
        if total <= budget:
            chosen = list(range(len(lines)))
        else:
            stride, position, chosen = total / max(budget, 1), 0.0, []
            while int(position) < len(lines):
                chosen.append(int(position))
                position += stride
            used, kept = 0, []
            for i in chosen:
                if used + costs[i] > budget:
                    break
                kept.append(i)
                used += costs[i]
            chosen = kept
        label = f"LOG EXCERPT ({len(chosen)} of {len(lines)} lines, sampled uniformly across the file)"
        return (label, "\n".join(lines[i] for i in chosen), [events[i]["line_no"] for i in chosen],
                {"context_line_fraction": round(len(chosen) / max(1, len(lines)), 5)})

    return _single_turn("long_context_uniform", backend, f"{GENERIC_RAG_PROMPT}\n\n{ANSWER_FORMAT_INSTRUCTION}",
                        question, build, thinking, started)


def _pack_chunks(results, budget_tokens: int):
    kept, used = [], 0
    for chunk, score in results:
        block = f"--- excerpt (type={chunk['chunk_type']}, score={score:.3f}) ---\n{chunk['text']}"
        cost = estimate_tokens(block)
        if used + cost > budget_tokens:
            continue
        kept.append((chunk, block))
        used += cost
    return kept


def _rag_builder(results):
    def build(budget):
        kept = _pack_chunks(results, budget)
        return ("RETRIEVED LOG EXCERPTS", "\n".join(b for _, b in kept), [n for c, _ in kept for n in c["line_numbers"]],
                {"chunks_used": len(kept), "chunk_types_used": [c["chunk_type"] for c, _ in kept]})
    return build


def answer_fixed_size_chunk_rag(workspace, backend, question: str, thinking: bool = False) -> dict:
    started = time.perf_counter()
    results = workspace.fixed_size_index.search(question, RAG_TOP_CHUNKS)
    return _single_turn("fixed_size_chunk_rag", backend, f"{GENERIC_RAG_PROMPT}\n\n{ANSWER_FORMAT_INSTRUCTION}",
                        question, _rag_builder(results), thinking, started)


def answer_multi_type_chunk_rag(workspace, backend, question: str, thinking: bool = False,
                                excluded_chunk_types=frozenset()) -> dict:
    started = time.perf_counter()
    results = workspace.multi_type_index.search(question, RAG_TOP_CHUNKS, excluded_types=excluded_chunk_types)
    return _single_turn("multi_type_chunk_rag", backend, final_system_prompt(workspace.characterization, "rag"),
                        question, _rag_builder(results), thinking, started)
