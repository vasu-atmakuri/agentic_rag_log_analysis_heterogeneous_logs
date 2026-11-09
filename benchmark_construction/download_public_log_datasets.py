"""Downloads the public LogHub datasets (Zookeeper, Linux) from Zenodo and the sentence embedding
model, so that every later step runs fully offline.

LogHub: J. Zhu et al., "Loghub: A Large Collection of System Log Datasets for AI-driven Log
Analytics," ISSRE 2023. Zenodo concept DOI 10.5281/zenodo.1144100.

Usage: python -m benchmark_construction.download_public_log_datasets --target-dir log_datasets
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import tarfile
import zipfile

import httpx

ZENODO_RECORD_CANDIDATES = [
    "https://zenodo.org/api/records/8196385",
    "https://zenodo.org/api/records/1144100/versions/latest",
]
WANTED = {"Zookeeper": "loghub_zookeeper", "Linux": "loghub_linux"}


def list_zenodo_files(client: httpx.Client) -> tuple[str, list[dict]]:
    errors = []
    for url in ZENODO_RECORD_CANDIDATES:
        try:
            response = client.get(url)
            response.raise_for_status()
            record = response.json()
            files = record.get("files") or []
            if files:
                return record.get("links", {}).get("self_html", url), files
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    raise SystemExit("Could not list LogHub files on Zenodo:\n  " + "\n  ".join(errors) +
                     "\nDownload Zookeeper.tar.gz and Linux.tar.gz manually from https://zenodo.org/records/8196385 "
                     "and place the .log files as log_datasets/loghub_zookeeper/Zookeeper.log and "
                     "log_datasets/loghub_linux/Linux.log")


def extract_log(archive_bytes: bytes, file_key: str, wanted_name: str) -> bytes:
    if file_key.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".log")]
            members.sort(key=lambda m: (os.path.basename(m.name) != f"{wanted_name}.log", -m.size))
            return tar.extractfile(members[0]).read()
    if file_key.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".log")]
            names.sort(key=lambda n: (os.path.basename(n) != f"{wanted_name}.log", -zf.getinfo(n).file_size))
            return zf.read(names[0])
    return archive_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", default="log_datasets")
    parser.add_argument("--skip-embedding-model", action="store_true")
    args = parser.parse_args()
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        page, files = list_zenodo_files(client)
        print(f"LogHub record: {page}")
        for name, folder in WANTED.items():
            out_dir = os.path.join(args.target_dir, folder)
            out_path = os.path.join(out_dir, f"{name}.log")
            if os.path.exists(out_path):
                print(f"{name}: already present ({out_path})")
                continue
            match = [f for f in files if f.get("key", "").split(".")[0].lower() == name.lower()]
            if not match:
                raise SystemExit(f"{name} archive not found in the Zenodo record; available: {[f.get('key') for f in files]}")
            entry = match[0]
            url = entry["links"].get("self") or entry["links"].get("content")
            print(f"{name}: downloading {entry['key']} ({entry.get('size', 0) / 1e6:.1f} MB)")
            data = client.get(url).content
            expected = (entry.get("checksum") or "").replace("md5:", "")
            if expected and hashlib.md5(data).hexdigest() != expected:
                raise SystemExit(f"{name}: checksum mismatch, download corrupted; rerun")
            log_bytes = extract_log(data, entry["key"], name)
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "wb") as handle:
                handle.write(log_bytes)
            line_count = log_bytes.count(b"\n")
            print(f"{name}: {line_count} lines -> {out_path}")
    if not args.skip_embedding_model:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(["warm-up"])
        print("Embedding model all-MiniLM-L6-v2 cached locally.")


if __name__ == "__main__":
    main()
