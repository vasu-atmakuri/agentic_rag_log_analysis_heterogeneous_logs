"""Code-writing agent baseline: the model answers by writing Python (pandas) that runs over a read-only table of
parsed events, instead of calling fixed tools.

The table `events` has columns: line (int), time (pandas UTC timestamp), timestamp_text (str, as in the log),
level (str), component (str), message (str). Code runs in a separate Python process with a 20-second limit,
an audit hook that blocks network access, subprocesses and file writes, and only its printed output (first
3,000 characters), plus the value of a final expression as in a notebook, is returned to the model. At most 10 model turns, as for the tool-using agent.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import time

from .agentic_tool_use_loop import MAX_ITERATIONS, _elide_all_but_last_observation, _fit_to_budget
from .local_language_model_backend import ContextLengthExceeded
from .log_source_characterization import ANSWER_FORMAT_INSTRUCTION
from .question_answering_methods import UsageTally, ensure_final_answer, extract_final_answer

CODE_TIMEOUT_SECONDS = 20
MAX_OUTPUT_CHARS = 3000

RUN_PYTHON_TOOL = [{"type": "function", "function": {
    "name": "run_python",
    "description": ("Run Python code over the log. A pandas DataFrame named `events` is preloaded with columns "
                    "line, time (UTC pandas Timestamp), timestamp_text, level, component, message. pandas is "
                    "available as pd and re as re. Printed output is returned, and the value of a final expression is "
                    "shown as in a notebook."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]

SYSTEM_PROMPT = ("You are a log analysis assistant. You can run Python code over the parsed log with the run_python "
                 "tool. Use it to compute exact answers (filters, counts, first and last occurrences) before answering "
                 "the user's question.")

CHILD_RUNNER = r'''
import sys, io, pickle, re, ast, contextlib
import pandas as pd
pd.set_option("display.max_rows", 40)
pd.set_option("display.width", 160)
def _guard(event, args):
    blocked = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.fork", "os.remove", "os.unlink",
               "os.rename", "os.rmdir", "shutil.", "os.mkdir", "os.chmod", "ctypes.")
    if event.startswith(blocked):
        raise PermissionError(f"blocked: {event}")
    if event == "open" and len(args) > 1 and isinstance(args[1], str) and any(m in args[1] for m in "wax+"):
        raise PermissionError("file writes are blocked")
events = pickle.load(open(sys.argv[1], "rb"))
code = sys.stdin.read()
sys.addaudithook(_guard)
buffer = io.StringIO()
try:
    namespace = {"events": events, "pd": pd, "re": re, "__name__": "__main__"}
    tree = ast.parse(code, "<model_code>", "exec")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    with contextlib.redirect_stdout(buffer):
        exec(compile(tree, "<model_code>", "exec"), namespace)
        if last is not None:  # notebook behaviour: the value of a final expression is shown
            value = eval(compile(ast.Expression(last.value), "<model_code>", "eval"), namespace)
            if value is not None:
                print(repr(value))
    out = buffer.getvalue()
except Exception as exc:
    out = buffer.getvalue() + f"\nERROR: {type(exc).__name__}: {exc}"
sys.stdout.write(out if out.strip() else "(no output; print the values you need)")
'''


def events_table_path(workspace) -> str:
    """Builds (once per process and workspace) a pickled DataFrame of primary events."""
    import pandas as pd

    cache = getattr(workspace, "_code_agent_table_path", None)
    if cache and os.path.exists(cache):
        return cache
    rows = [{"line": e["line_no"], "time": pd.to_datetime(e["epoch"], unit="s", utc=True) if e["epoch"] is not None else pd.NaT,
             "timestamp_text": e["ts_str"], "level": e["level"], "component": e["logger"], "message": e["message"]}
            for e in workspace.primary]
    frame = pd.DataFrame(rows)
    handle, path = tempfile.mkstemp(prefix=f"events_{workspace.dataset_name}_", suffix=".pkl")
    with os.fdopen(handle, "wb") as out:
        pickle.dump(frame, out)
    workspace._code_agent_table_path = path
    return path


def run_model_code(table_path: str, code: str) -> str:
    try:
        completed = subprocess.run([sys.executable, "-I", "-c", CHILD_RUNNER, table_path], input=code,
                                   capture_output=True, text=True, timeout=CODE_TIMEOUT_SECONDS)
        output = completed.stdout or ""
        if completed.returncode != 0 and completed.stderr:
            output += "\nERROR: " + completed.stderr.strip().splitlines()[-1]
    except subprocess.TimeoutExpired:
        output = f"ERROR: code did not finish within {CODE_TIMEOUT_SECONDS} seconds"
    return output if len(output) <= MAX_OUTPUT_CHARS else output[:MAX_OUTPUT_CHARS] + "\n...[output truncated]"


def answer_code_writing_agent(workspace, backend, question: str, thinking: bool = False, **_) -> dict:
    started = time.perf_counter()
    table = events_table_path(workspace)
    s = workspace.stats
    system = (f"{SYSTEM_PROMPT}\n\nThe log covers {s['first_timestamp']} to {s['last_timestamp']}.\n\n"
              f"{ANSWER_FORMAT_INSTRUCTION}")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"QUESTION: {question}"}]
    budget = backend.prompt_budget_tokens(thinking) - 400
    tally = UsageTally()
    trace, code_runs, elided, errors = [], 0, 0, 0
    final_text, finished, last_response = "", False, None

    def call(msgs, with_tools):
        nonlocal elided
        for attempt in range(3):
            try:
                return backend.chat(msgs, tools=RUN_PYTHON_TOOL if with_tools else None, thinking=thinking)
            except ContextLengthExceeded:
                tally.context_retries += 1
                elided += _elide_all_but_last_observation(msgs)
        raise ContextLengthExceeded("code agent conversation does not fit the context window")

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
            code = str((args or {}).get("code", "")) if name == "run_python" else ""
            observation = run_model_code(table, code) if code else f"Unknown tool or empty code for '{name}'."
            code_runs += 1
            errors += "ERROR:" in observation
            trace.append({"iteration": iteration, "type": "tool", "tool": "run_python", "arguments": {"code": code[:1500]},
                          "observation_chars": len(observation), "observation_head": observation[:400]})
            messages.append({"role": "tool", "content": observation, "tool_name": "run_python"})
    if not finished:
        messages.append({"role": "user", "content": "Tool budget exhausted. Give your final answer now from the evidence gathered."})
        elided += _fit_to_budget(messages, budget + 400)
        last_response = call(messages, False)
        tally.add(last_response)
        final_text = last_response.content
        trace.append({"iteration": MAX_ITERATIONS, "type": "forced_final", "content": final_text[:2000]})
    final_text, extracted = ensure_final_answer(backend, messages, last_response, tally, thinking)
    return {
        "method": "code_writing_agent", "response_text": final_text, "final_answer": extract_final_answer(final_text),
        "answer_extraction_followup": extracted, "thinking_chars": tally.thinking_chars, "llm_calls": tally.llm_calls,
        "tool_calls": code_runs, "code_errors": errors, "prompt_tokens": tally.prompt_tokens,
        "completion_tokens": tally.completion_tokens, "llm_seconds": round(tally.llm_seconds, 3),
        "wall_seconds": round(time.perf_counter() - started, 3), "context_overflow_risk": tally.overflow,
        "output_truncated": tally.truncated, "context_retries": tally.context_retries, "elided_observations": elided,
        "hit_iteration_limit": not finished, "evidence_line_numbers": [], "tools_used": ["run_python"] * code_runs,
        "trace": trace,
    }
