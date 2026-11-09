"""Checks how a local model behaves under the harness settings: whether reasoning text leaks into
the answer, whether output is cut off, whether tools are called, and how many tokens log text uses.

Usage: python scripts/diagnose_local_model_behaviour.py qwen3:4b [other models...]
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama  # noqa: E402

from log_analysis_system.agentic_tool_use_loop import TOOL_DEFINITIONS  # noqa: E402
from log_analysis_system.log_source_characterization import ANSWER_FORMAT_INSTRUCTION  # noqa: E402

LOG = "log_datasets/fault_injected_inference_service/fault_injected_inference_service.log"
QUESTION = "Which component first reported the problem that led to the burst of HTTP 503 responses around 2026-03-09 14:38?"
EXCERPT = """2026-03-09 14:37:10,000 - db.connector - WARN - connection pool usage 19/20 active, wait queue 4
2026-03-09 14:37:30,000 - db.connector - WARN - connection pool usage 20/20 active, wait queue 17
2026-03-09 14:37:55,216 - db.connector - ERROR - timeout acquiring connection from pool after 30000 ms
2026-03-09 14:38:12,611 - storage.feature_store - ERROR - feature lookup failed for request req-1: database timeout
2026-03-09 14:38:20,104 - api.gateway - ERROR - POST /v1/predict status=503 latency_ms=30012 request_id=req-2"""


def show(label, r, start):
    m = r.message
    content = m.content or ""
    print(f"  [{label}] {time.perf_counter() - start:5.1f}s done_reason={getattr(r, 'done_reason', None)} "
          f"prompt_tokens={r.prompt_eval_count} output_tokens={r.eval_count} thinking_chars={len(getattr(m, 'thinking', '') or '')} "
          f"has_think_tag={'<think>' in content} tool_calls={[(c.function.name, dict(c.function.arguments)) for c in (m.tool_calls or [])]}")
    print("      content start:", repr(content[:220]))
    print("      content end:  ", repr(content[-160:]))


def main():
    models = sys.argv[1:] or ["qwen3:4b"]
    lines = open(LOG, encoding="utf-8").read().splitlines()
    for model in models:
        print(f"\n===== {model}")
        opts = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 512}
        msgs = [{"role": "system", "content": "Answer from the log excerpt.\n\n" + ANSWER_FORMAT_INSTRUCTION},
                {"role": "user", "content": f"LOG EXCERPT\n{EXCERPT}\n\nQUESTION: {QUESTION}"}]
        for think in (False, True):
            t = time.perf_counter()
            show(f"answer think={think}", ollama.chat(model=model, messages=msgs, options=opts, think=think), t)
        t = time.perf_counter()
        agent_msgs = [{"role": "system", "content": "You are a log analyst with tools. Use the tools to gather evidence.\n\n" + ANSWER_FORMAT_INSTRUCTION},
                      {"role": "user", "content": "QUESTION: How many ERROR-level events were logged between 2026-03-09 14:00 and 2026-03-09 15:00?"}]
        show("tool call think=False", ollama.chat(model=model, messages=agent_msgs, tools=TOOL_DEFINITIONS, options=opts, think=False), t)
        for n in (100, 250, 400):
            nonce = f"run {random.randint(0, 10**9)}\n"
            block = nonce + "\n".join(lines[5000:5000 + n])
            t = time.perf_counter()
            r = ollama.chat(model=model, messages=[{"role": "user", "content": block + "\n\nReply with the word OK."}],
                            options={**opts, "num_predict": 8}, think=False)
            print(f"  [long prompt {n} lines] chars={len(block)} prompt_tokens={r.prompt_eval_count} "
                  f"chars_per_token={len(block) / max(1, r.prompt_eval_count):.2f} {time.perf_counter() - t:5.1f}s")


if __name__ == "__main__":
    main()
