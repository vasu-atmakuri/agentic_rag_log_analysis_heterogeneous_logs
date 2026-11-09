"""Downloads the Loghub-2.0 full datasets used in the extended study and lists the RCAEval files.

Loghub-2.0: Z. Jiang et al., "A Large-scale Evaluation for Log Parsing Techniques: How Far are We?", ISSTA 2024.
Zenodo record 8275861. Each dataset provides the raw log (system input) and a structured CSV annotated by the
dataset authors (used only to compute gold answers, independently of the system's parser).
RCAEval: L. Pham et al., Zenodo record 14504481 (root cause analysis of microservice systems).

Usage: python -m benchmark_construction.download_loghub2_and_rcaeval --target-dir log_datasets/loghub2
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import zipfile

import httpx

LOGHUB2_RECORD = "https://zenodo.org/api/records/8275861"
RCAEVAL_RECORDS = ["https://zenodo.org/api/records/14504481", "https://zenodo.org/api/records/14504480"]
WANTED = ["Apache", "Hadoop", "HealthApp", "HPC", "Linux", "Mac", "OpenSSH", "OpenStack", "Proxifier", "Zookeeper"]


def files_of(client, url):
    response = client.get(url)
    response.raise_for_status()
    record = response.json()
    return record.get("files") or [], record.get("metadata", {}).get("title", "")


def extract(data: bytes, key: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if key.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if member.endswith("/") or "__MACOSX" in member:
                    continue
                target = os.path.join(out_dir, os.path.basename(member))
                with open(target, "wb") as handle:
                    handle.write(zf.read(member))
    elif key.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    with open(os.path.join(out_dir, os.path.basename(member.name)), "wb") as handle:
                        handle.write(tar.extractfile(member).read())
    else:
        with open(os.path.join(out_dir, key), "wb") as handle:
            handle.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", default="log_datasets/loghub2")
    args = parser.parse_args()
    report = {"loghub2": {}, "rcaeval_files": []}
    with httpx.Client(follow_redirects=True, timeout=1800) as client:
        files, title = files_of(client, LOGHUB2_RECORD)
        print(f"Loghub-2.0 record: {title}; {len(files)} files")
        for f in files:
            print(f'  {f["key"]}  {f.get("size", 0) / 1e6:.1f} MB')
        for name in WANTED:
            out_dir = os.path.join(args.target_dir, name)
            if os.path.isdir(out_dir) and any(p.endswith(".log") for p in os.listdir(out_dir)):
                print(f"{name}: already present")
            else:
                match = [f for f in files if f["key"].split(".")[0].lower() == name.lower()]
                if not match:
                    print(f"{name}: NOT FOUND in record")
                    continue
                entry = match[0]
                url = entry["links"].get("self") or entry["links"].get("content")
                print(f"{name}: downloading {entry['key']} ({entry.get('size', 0) / 1e6:.1f} MB)")
                data = client.get(url).content
                expected = (entry.get("checksum") or "").replace("md5:", "")
                if expected and hashlib.md5(data).hexdigest() != expected:
                    print(f"{name}: CHECKSUM MISMATCH, skipped")
                    continue
                extract(data, entry["key"], out_dir)
            listing = {}
            for p in sorted(os.listdir(out_dir)):
                path = os.path.join(out_dir, p)
                with open(path, "rb") as handle:
                    head = handle.read(600).decode("utf-8", "replace").splitlines()[:3]
                lines = sum(1 for _ in open(path, "rb"))
                listing[p] = {"bytes": os.path.getsize(path), "lines": lines, "head": head}
            report["loghub2"][name] = listing
        for url in RCAEVAL_RECORDS:
            try:
                rfiles, rtitle = files_of(client, url)
                report["rcaeval_files"] = [{"key": f["key"], "mb": round(f.get("size", 0) / 1e6, 1)} for f in rfiles]
                report["rcaeval_title"] = rtitle
                break
            except Exception as exc:  # noqa: BLE001
                report["rcaeval_error"] = str(exc)
    os.makedirs(args.target_dir, exist_ok=True)
    with open(os.path.join(args.target_dir, "download_inspection_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print("Inspection report written.")


if __name__ == "__main__":
    main()
