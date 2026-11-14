"""Runs the frozen experiment matrix and appends one JSON record per (question, method, condition).

Resumable: completed runs are skipped, so the process can be stopped (Ctrl+C) and restarted at any
time. Methods are interleaved per question so partial results stay balanced across methods.

  --pace day    lowers this process's CPU priority and pauses between runs (use the Mac meanwhile)
  --pace night  runs at full speed

Usage:
  python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_journal_extended.yaml --pace night
  python -m experiment_evaluation.run_experiment_matrix --config configs/experiment_matrix_journal_extended.yaml --only main_comparison__qwen3_4b --split dev --limit 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # embedding model is cached; never wait on the network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from log_analysis_system import (ANSWER_METHODS, METHODS_SUPPORTING_CHUNK_ABLATION,  # noqa: E402
                                 METHODS_SUPPORTING_TOOL_ABLATION, METHODS_WITHOUT_EVIDENCE_TRACKING)
from log_analysis_system.local_language_model_backend import ModelRuntimeSettings, build_backend  # noqa: E402
from log_analysis_system.log_analysis_workspace import prepare_workspace  # noqa: E402
from experiment_evaluation.automatic_answer_grading import CHECKER_VERSION, UnsupportedClaimChecker, grade_final_answer  # noqa: E402


def run_key(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]


def load_questions(benchmark_dir: str, dataset: str, split: str, variant: str | None = None) -> list[dict]:
    suffix = f"__{variant}" if variant else ""
    path = os.path.join(benchmark_dir, f"{dataset}__{split}_questions{suffix}.jsonl")
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_keys(path: str) -> set[str]:
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "ok":
                    done.add(record["run_key"])
    return done


def evidence_recall(question: dict, retrieved: list[int]):
    key = question.get("key_evidence_line_numbers") or []
    if not key:
        return None
    got = set(retrieved)
    return round(sum(1 for n in key if n in got) / len(key), 4)


def write_environment_record(results_dir: str, settings_by_model: dict, scripted: bool):
    info = {"recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "python": sys.version,
            "platform": platform.platform(), "machine": platform.machine(), "models": {}}
    for pkg in ("faiss", "sentence_transformers", "rank_bm25", "ollama", "numpy", "torch", "transformers"):
        try:
            module = __import__(pkg)
            info[f"version_{pkg}"] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            info[f"version_{pkg}"] = "not installed"
    if not scripted:
        try:
            import ollama

            for model in settings_by_model:
                shown = ollama.show(model)
                details = getattr(shown, "details", None)
                info["models"][model] = {
                    "quantization": getattr(details, "quantization_level", None),
                    "parameter_size": getattr(details, "parameter_size", None),
                    "family": getattr(details, "family", None),
                    "modified_at": str(getattr(shown, "modified_at", "")),
                }
            listed = ollama.list()
            for entry in getattr(listed, "models", []):
                name = getattr(entry, "model", None)
                if name in info["models"]:
                    info["models"][name]["digest"] = getattr(entry, "digest", None)
        except Exception as exc:  # noqa: BLE001
            info["model_query_error"] = str(exc)
    with open(os.path.join(results_dir, "run_environment.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


MODEL_SERVER_URL = "http://127.0.0.1:11434/api/version"
CONNECTION_FAILURE_MARKERS = ("ConnectionError", "RemoteProtocolError", "Server disconnected", "Connection refused",
                              "ReadTimeout", "timed out")
HUNG_SERVER_MARKERS = ("ReadTimeout", "timed out")


def model_server_reachable() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(MODEL_SERVER_URL, timeout=5) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_for_model_server(scripted: bool, pace: str, max_wait_seconds: int = 900, force_restart: bool = False) -> bool:
    """If the local model server is down (or hung, force_restart), restart it and wait until it answers."""
    if scripted or (model_server_reachable() and not force_restart):
        return True
    print(f"\nModel server unreachable or hung at {datetime.now(timezone.utc).isoformat(timespec='seconds')}; restarting it.")
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "switch_model_server_pace.sh")
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        subprocess.run([script, pace], capture_output=True, text=True)
        for _ in range(12):
            if model_server_reachable():
                print("Model server back.")
                return True
            time.sleep(5)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", nargs="*", help="experiment names to run (default: all, in file order)")
    parser.add_argument("--split", default="test", choices=["test", "dev"])
    parser.add_argument("--limit", type=int, default=None, help="questions per dataset (smoke tests)")
    parser.add_argument("--pace", default="night", choices=["day", "night"])
    parser.add_argument("--scripted-backend", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--hashing-embedder", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(cfg["datasets_config"], "r", encoding="utf-8") as handle:
        datasets_cfg = yaml.safe_load(handle)["datasets"]
    if args.pace == "day":
        try:
            os.nice(15)
        except OSError:
            pass
    pause = 1.5 if args.pace == "day" else 0.0

    experiments = [e for e in cfg["experiments"] if not args.only or e["name"] in args.only]
    results_dir = cfg["results_dir"] if args.split == "test" else os.path.join(cfg["results_dir"], "dev_smoke_tests")
    os.makedirs(os.path.join(results_dir, "raw_runs"), exist_ok=True)
    runtime = cfg.get("runtime", {})
    settings_by_model = {e["model"]: ModelRuntimeSettings(model=e["model"], **runtime) for e in experiments}
    settings_by_experiment = {e["name"]: ModelRuntimeSettings(model=e["model"], **{**runtime, **e.get("runtime_overrides", {})})
                              for e in experiments}
    write_environment_record(results_dir, settings_by_model, args.scripted_backend)
    embedder_kind = "hashing_test" if args.hashing_embedder else "sentence_transformer"

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    for exp in experiments:
        backend = build_backend(settings_by_experiment[exp["name"]], scripted=args.scripted_backend)
        disabled_tools = frozenset(exp.get("disabled_tools", []))
        variant = exp.get("question_variant")
        repeats = int(exp.get("repeats", 1))
        out_path = os.path.join(results_dir, "raw_runs", f'{exp["name"]}.jsonl')
        done = completed_keys(out_path)
        dataset_names = list(datasets_cfg) if exp.get("datasets", "all") == "all" else exp["datasets"]
        excluded = frozenset(exp.get("excluded_chunk_types", []))
        thinking = bool(exp.get("thinking", False))
        condition = exp.get("characterization", "automatic")
        for ds in dataset_names:
            spec = datasets_cfg[ds]
            if not load_questions(cfg["benchmark_dir"], ds, args.split, variant):
                continue  # no questions for this source in this split (e.g. RCAEval cases held out for development)
            expert_path = os.path.join(cfg["expert_prompts_dir"], f"{ds}.yaml")
            try:
                workspace = prepare_workspace(ds, spec["log_path"], cfg["artifacts_dir"], condition, backend=backend,
                                              expert_config_path=expert_path, embedder_kind=embedder_kind,
                                              base_year_for_yearless=spec.get("base_year", 2005),
                                              characterization_model_tag=exp["model"])
            except (FileNotFoundError, ValueError) as exc:
                print(f'\nSKIPPED {exp["name"]} / {ds}: {exc}')
                continue
            checker = UnsupportedClaimChecker(workspace)
            questions = load_questions(cfg["benchmark_dir"], ds, args.split, variant)
            subset = set(exp.get("question_categories") or []) or None
            if subset:
                questions = [q for q in questions if q["category"] in subset]
            if exp.get("max_questions_per_dataset"):
                cap = exp["max_questions_per_dataset"]
                step = max(1, -(-len(questions) // cap))
                questions = questions[::step][:cap]  # evenly spread across categories
            if args.limit:
                questions = questions[: args.limit]
            def key_for(q, m, r):
                parts = [exp["name"], ds, q["question_id"], m, exp["model"], condition, thinking, sorted(excluded)]
                if r:
                    parts.append(f"repeat{r}")
                if disabled_tools:
                    parts.append("disabled:" + ",".join(sorted(disabled_tools)))
                if variant:
                    parts.append(f"variant:{variant}")
                return run_key(*parts)

            jobs = [(q, m, r) for r in range(repeats) for q in questions for m in exp["methods"]]
            pending = [(q, m, r) for q, m, r in jobs if key_for(q, m, r) not in done]
            label = f'{exp["name"]} / {ds}'
            print(f"\n{label}: {len(jobs) - len(pending)} done, {len(pending)} to run")
            iterator = tqdm(pending, desc=label[:60], unit="run") if tqdm else pending
            with open(out_path, "a", encoding="utf-8") as sink:
                for question, method, repeat_index in iterator:
                    key = key_for(question, method, repeat_index)
                    record = {"run_key": key, "experiment": exp["name"], "dataset": ds,
                              "question_id": question["question_id"], "category": question["category"],
                              "template": question["template"], "answer_type": question["answer_type"],
                              "method": method, "model": exp["model"], "characterization": condition,
                              "thinking": thinking, "excluded_chunk_types": sorted(excluded),
                              "repeat_index": repeat_index, "disabled_tools": sorted(disabled_tools),
                              "question_variant": variant, "num_ctx": backend.settings.num_ctx,
                              "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                    hung = False
                    for attempt in (1, 2):
                        wait_for_model_server(args.scripted_backend, args.pace, force_restart=hung)
                        wall_start, mono_start = time.time(), time.monotonic()
                        try:
                            fn = ANSWER_METHODS[method]
                            kwargs = {"thinking": thinking}
                            if method in METHODS_SUPPORTING_CHUNK_ABLATION:
                                kwargs["excluded_chunk_types"] = excluded
                            if method in METHODS_SUPPORTING_TOOL_ABLATION and disabled_tools:
                                kwargs["disabled_tools"] = disabled_tools
                            result = fn(workspace, backend, question["question"], **kwargs)
                            grade = grade_final_answer(question, result["final_answer"])
                            claims = checker.check(result["response_text"], question["question"])
                            trace = result.pop("trace", [])
                            evidence = result.pop("evidence_line_numbers", [])
                            result["response_text"] = result["response_text"][:4000]
                            implementation = result.pop("method", method)
                            record.update(result)
                            record["method"] = method  # the configured method name (e.g. a parameterized variant)
                            record["implementation"] = implementation
                            record.update(grade)
                            record.update(claims)
                            record["key_evidence_recall"] = (None if method in METHODS_WITHOUT_EVIDENCE_TRACKING
                                                             else evidence_recall(question, evidence))
                            record["evidence_lines_retrieved"] = len(evidence)
                            record["trace"] = trace
                            record["claim_checker_version"] = CHECKER_VERSION
                            record["status"] = "ok"
                        except KeyboardInterrupt:
                            print("\nStopped by user. Completed runs are saved; rerun the same command to resume.")
                            return
                        except Exception as exc:  # noqa: BLE001
                            record["status"] = "error"
                            record["error"] = f"{type(exc).__name__}: {exc}"
                            record["traceback"] = traceback.format_exc()[-2000:]
                        # wall time minus monotonic time: seconds the Mac was asleep during this run (macOS)
                        record["suspended_seconds"] = round(max(0.0, (time.time() - wall_start) - (time.monotonic() - mono_start)), 1)
                        retry = record["status"] == "error" and any(m in record["error"] for m in CONNECTION_FAILURE_MARKERS)
                        if not retry or attempt == 2:
                            break
                        hung = any(m in record["error"] for m in HUNG_SERVER_MARKERS)
                        record.pop("error", None)
                        record.pop("traceback", None)
                    sink.write(json.dumps(record) + "\n")
                    sink.flush()
                    if pause:
                        time.sleep(pause)
    print("\nAll requested experiments finished.")


if __name__ == "__main__":
    main()
