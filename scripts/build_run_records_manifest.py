"""Builds the manifest of all run records: one row per records file with its study component, number of runs,
runs that ended in an error, models, methods and SHA-256, and a summary of the run totals reported in the papers.

Usage: python scripts/build_run_records_manifest.py   (writes run_records_manifest.csv and RUN_RECORDS_MANIFEST.md)
"""
import csv
import glob
import hashlib
import json
import os
from collections import Counter

COMPONENTS = {
    "experiment_results_journal_extended": "Template benchmark (626 questions): main comparison, ablations, characterization, repeats, paraphrases, 32K context",
    "experiment_results_revision_experiments": "Template benchmark, added in review: BM25 with 20 and 50 lines, single-shot query, timeline ablation with Qwen3-30B-A3B",
    "experiment_results_parser_revision_2_sensitivity": "Template benchmark, added in review: agents with the corrected parser on HealthApp, Mac and HPC",
    "experiment_results_independent_query_samples": "Template benchmark, added in review: three sampled single-shot queries per question",
    "experiment_results_engineer_written": "Practitioner-written questions (50): seven methods",
    "experiment_results_revision_experiments_practitioner": "Practitioner-written questions, added in review: BM25 with 20 and 50 lines, single-shot query",
    "experiment_results_rcaeval_re3": "RCAEval RE3 logs-only root-cause cases (84)",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    rows, totals = [], Counter()
    for folder in sorted(glob.glob("experiment_results_*")):
        for sub in ("raw_runs",) + tuple(sorted(os.path.basename(p) for p in glob.glob(os.path.join(folder, "archived_*")))):
            for path in sorted(glob.glob(os.path.join(folder, sub, "*.jsonl"))):
                records, last = 0, {}
                models, methods = set(), set()
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        r = json.loads(line)
                        records += 1
                        last[r["run_key"]] = r  # a run that ended in an error and was repeated keeps its last record
                        models.add(r.get("model"))
                        methods.add(r.get("method"))
                runs = len(last)
                errors = sum(1 for r in last.values() if r.get("status") == "error")
                kind = "included" if sub == "raw_runs" else "excluded (archived, see study design)"
                rows.append({"component": COMPONENTS.get(folder, folder), "folder": folder, "records_file": path,
                             "status": kind, "records": records, "runs": runs, "superseded_error_records": records - runs, "runs_ending_in_error": errors,
                             "models": ";".join(sorted(m for m in models if m)),
                             "methods": ";".join(sorted(m for m in methods if m)), "sha256": sha256(path)})
                if sub == "raw_runs":
                    totals[folder] += runs
    with open("run_records_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open("RUN_RECORDS_MANIFEST.md", "w", encoding="utf-8") as out:
        out.write("# Run records manifest\n\nOne JSON line per run (question, method, model, prompt settings, model output, "
                  "tool trace, timing, grade). A run that ended in an error and was repeated by the resumable runner "
                  "keeps both records; the analysis uses the last record of each run key, and the counts below are "
                  "unique runs. Runs whose last record is an error are counted as incorrect. Per-file counts and SHA-256 hashes are in `run_records_manifest.csv`.\n\n")
        out.write("| Results folder | Study component | Included runs |\n|---|---|---:|\n")
        for folder, n in totals.items():
            out.write(f"| `{folder}` | {COMPONENTS.get(folder, folder)} | {n:,} |\n")
        template = sum(totals[f] for f in ("experiment_results_journal_extended", "experiment_results_revision_experiments",
                                           "experiment_results_parser_revision_2_sensitivity",
                                           "experiment_results_independent_query_samples"))
        out.write(f"\nTemplate benchmark, all experiments: {template:,} runs. Practitioner-written questions: "
                  f"{totals['experiment_results_engineer_written'] + totals['experiment_results_revision_experiments_practitioner']:,} runs. "
                  f"RCAEval RE3: {totals['experiment_results_rcaeval_re3']:,} runs.\n")
    print(dict(totals))


if __name__ == "__main__":
    main()
