"""Downloads RCAEval RE3 (code-level faults with logs and labelled root causes) and writes an inspection report.

RCAEval: L. Pham et al., "RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry
Data", Zenodo record 14504481.

Usage: python -m benchmark_construction.download_rcaeval_re3 --target-dir log_datasets/rcaeval
"""
import argparse
import json
import os
import zipfile

import httpx

RECORD = "https://zenodo.org/api/records/14504481"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", default="log_datasets/rcaeval")
    parser.add_argument("--archive", default="RE3.zip")
    args = parser.parse_args()
    os.makedirs(args.target_dir, exist_ok=True)
    zip_path = os.path.join(args.target_dir, args.archive)
    if not os.path.exists(zip_path):
        with httpx.Client(follow_redirects=True, timeout=3600) as client:
            files = client.get(RECORD).json()["files"]
            entry = [f for f in files if f["key"] == args.archive][0]
            url = entry["links"].get("self") or entry["links"].get("content")
            with client.stream("GET", url) as response, open(zip_path + ".part", "wb") as out:
                for chunk in response.iter_bytes(1 << 20):
                    out.write(chunk)
            os.replace(zip_path + ".part", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(args.target_dir)
    report = {"members": len(names), "first_members": names[:60], "extensions": {}}
    for n in names:
        ext = os.path.splitext(n)[1] or "(dir/none)"
        report["extensions"][ext] = report["extensions"].get(ext, 0) + 1
    samples = {}
    for n in names:
        if n.endswith((".csv", ".json", ".txt", ".log", ".md")) and len(samples) < 12:
            path = os.path.join(args.target_dir, n)
            if os.path.isfile(path):
                with open(path, "rb") as handle:
                    samples[n] = handle.read(800).decode("utf-8", "replace").splitlines()[:4]
    report["samples"] = samples
    with open(os.path.join(args.target_dir, "rcaeval_re3_inspection_report.json"), "w") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps({k: report[k] for k in ("members", "extensions")}, indent=1))


if __name__ == "__main__":
    main()
