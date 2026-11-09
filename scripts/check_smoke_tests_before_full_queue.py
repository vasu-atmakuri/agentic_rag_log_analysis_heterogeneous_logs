"""Gate before the full extended queue: both development smoke tests must have produced records for every
method with no unresolved errors. Prints a per-method summary and exits non-zero if the gate fails."""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

CHECKS = {
    "journal extended": ("experiment_results_journal_extended/dev_smoke_tests/raw_runs", 7),
    "rcaeval re3": ("experiment_results_rcaeval_re3/dev_smoke_tests/raw_runs", 7),
}
failed = False
for label, (folder, n_methods) in CHECKS.items():
    status = defaultdict(Counter)
    correct = Counter()
    for path in glob.glob(os.path.join(folder, "*.jsonl")):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            status[r["method"]][r["status"]] += 1
            correct[r["method"]] += int(bool(r.get("correct")))
    print(f"\n== {label}")
    for method in sorted(status):
        print(f"  {method:26s} ok={status[method]['ok']:3d} error={status[method]['error']:3d} correct={correct[method]}")
    errors = sum(c["error"] for c in status.values())
    if len(status) < n_methods or errors or not all(c["ok"] for c in status.values()):
        print(f"  GATE FAILED: {len(status)} of {n_methods} methods present, {errors} errors")
        failed = True
sys.exit(1 if failed else 0)
