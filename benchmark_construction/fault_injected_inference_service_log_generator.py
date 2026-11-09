"""Synthetic but fully specified testbed: a seven-component ML inference service over three days,
written in Python logging format, with three service starts and seven injected incidents whose
causal chains are known exactly. The generator writes the log and a ground-truth manifest
(every event, every start, every incident) from which benchmark answers are computed, so gold
answers never depend on the system's own parser.

Usage: python -m benchmark_construction.fault_injected_inference_service_log_generator --output-dir DIR [--seed N]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timedelta, timezone

START = datetime(2026, 3, 9, 0, 0, 0, tzinfo=timezone.utc)
DURATION_DAYS = 3
LOG_FILE_NAME = "fault_injected_inference_service.log"
MANIFEST_FILE_NAME = "fault_injected_inference_service_ground_truth.json"

LOGGERS = ["api.gateway", "auth.service", "inference.scheduler", "inference.model_loader",
           "storage.feature_store", "db.connector", "metrics.exporter"]

SERVICE_STARTS = [
    {"start_id": "initial_start", "at": "2026-03-09 00:00:05", "model_version": "v3.2.1", "batch_size": 32,
     "workers": 8, "port": 8443, "load_seconds": 12.4},
    {"start_id": "redeploy", "at": "2026-03-10 10:15:00", "model_version": "v3.3.0", "batch_size": 64,
     "workers": 8, "port": 8443, "load_seconds": 13.1},
    {"start_id": "restart_after_crash", "at": "2026-03-11 03:31:00", "model_version": "v3.3.0", "batch_size": 32,
     "workers": 6, "port": 8443, "load_seconds": 12.9},
]
DB_HOST = "db-primary.internal:5432"
CACHE_HOST = "redis-cache-02.internal:6379"
GPU_DEVICE = "cuda:0"

INCIDENTS = [
    {"incident_id": "database_connection_pool_exhaustion", "at": "2026-03-09 14:37:10",
     "symptom": "the burst of HTTP 503 responses from the API gateway", "root_logger": "db.connector",
     "root_cause_keywords": ["connection pool", "pool exhaust", "acquiring connection", "pool usage", "database connection"],
     "resolution_keywords": ["pool usage 6/20", "pool recovered", "pool usage dropped", "pool returned to normal",
                             "pool usage normal", "connections released", "wait queue 0"]},
    {"incident_id": "identity_provider_certificate_expiry", "at": "2026-03-09 22:05:30",
     "symptom": "the wave of HTTP 401 unauthorized responses", "root_logger": "auth.service",
     "root_cause_keywords": ["certificate", "expired", "tls", "trust bundle"],
     "resolution_keywords": ["trust bundle", "reloaded", "certificate renewed", "new certificate", "valid until"]},
    {"incident_id": "feature_cache_connection_refused", "at": "2026-03-10 06:48:20",
     "symptom": "the HTTP 504 gateway timeouts", "root_logger": "storage.feature_store",
     "root_cause_keywords": ["redis", "connection refused", "feature cache", "cache unavailable", "redis-cache-02"],
     "resolution_keywords": ["redis connection restored", "connection restored", "cache restored", "reconnected"]},
    {"incident_id": "invalid_configuration_reload", "at": "2026-03-10 16:20:00",
     "symptom": "the failed model reload", "root_logger": "inference.model_loader",
     "root_cause_keywords": ["batch_size=-1", "invalid configuration", "invalid config", "batch_size", "invalid value"],
     "resolution_keywords": ["reloaded successfully", "configuration reloaded", "batch_size=64", "corrected configuration"]},
    {"incident_id": "gpu_out_of_memory", "at": "2026-03-11 03:22:00",
     "symptom": "the scheduler worker crashes", "root_logger": "inference.model_loader",
     "root_cause_keywords": ["out of memory", "oom", "gpu memory", "cuda out of memory", "memory exhaustion"],
     "resolution_keywords": ["restart", "restarted", "service restart", "starting inference service", "reduced batch"]},
    {"incident_id": "disk_full_on_shared_volume", "at": "2026-03-11 13:55:00",
     "symptom": "the database write-ahead log failures", "root_logger": "metrics.exporter",
     "root_cause_keywords": ["no space left", "disk full", "disk space", "out of disk", "disk pressure", "full disk"],
     "resolution_keywords": ["log rotation", "disk usage 71%", "rotation", "freed disk", "resuming spool"]},
    {"incident_id": "database_connection_pool_recurrence", "at": "2026-03-11 19:10:00",
     "symptom": "the second burst of HTTP 503 responses", "root_logger": "db.connector",
     "root_cause_keywords": ["connection pool", "acquiring connection", "pool usage"], "resolution_keywords": ["pool usage"],
     "used_for_causal_questions": False},
]


def ts(at: str) -> float:
    return datetime.strptime(at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def fmt(epoch: float) -> str:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{int(dt.microsecond / 1000):03d}"


class EventCollector:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.events = []

    def add(self, epoch, logger, level, message, continuation=None, tag=None):
        self.events.append({"epoch": round(epoch, 3), "logger": logger, "level": level, "message": message,
                            "continuation": continuation or [], "tag": tag})

    def request_id(self):
        return "req-" + "".join(self.rng.choice("0123456789abcdef") for _ in range(10))


def downtime_intervals():
    """Periods when the request path is not serving (before first start, redeploy gap, crash gap)."""
    return [(ts("2026-03-09 00:00:00"), ts("2026-03-09 00:00:25")),
            (ts("2026-03-10 10:14:40"), ts("2026-03-10 10:15:20")),
            (ts("2026-03-11 03:23:10"), ts("2026-03-11 03:31:20"))]


def in_downtime(epoch, intervals):
    return any(a <= epoch <= b for a, b in intervals)


def generate_background(c: EventCollector, end: float):
    rng = c.rng
    down = downtime_intervals()
    t0 = START.timestamp()
    t = t0 + 20
    while t < end:  # noisy metrics heartbeat
        c.add(t + rng.uniform(0, 0.4), "metrics.exporter", "INFO",
              f"heartbeat ok uptime_s={int(t - t0)} scrape_targets=7")
        if int(t - t0) % 60 < 15:
            c.add(t + 0.5 + rng.uniform(0, 0.3), "metrics.exporter", "INFO",
                  f"exported {rng.randint(380, 440)} metrics to prometheus-gw.internal:9091")
        t += 15
    t = t0 + 30
    while t < end:  # request traffic
        t += rng.expovariate(1 / 20)
        if in_downtime(t, down):
            continue
        rid = c.request_id()
        if rng.random() < 0.006:
            c.add(t, "api.gateway", "WARN", f"POST /v1/predict status=400 validation failed: missing field 'image' request_id={rid}")
        else:
            c.add(t, "api.gateway", "INFO", f"POST /v1/predict status=200 latency_ms={int(rng.gauss(38, 9)) + 5} request_id={rid}")
    t = t0 + 60
    while t < end:
        c.add(t + rng.uniform(0, 1), "auth.service", "INFO", f"validated {rng.randint(90, 180)} tokens in last 60s")
        if rng.random() < 0.004:
            c.add(t + 2, "auth.service", "WARN", f"rejected malformed bearer token from client 10.4.{rng.randint(0, 255)}.{rng.randint(1, 254)}")
        t += 60
    t = t0 + 45
    while t < end:
        if not in_downtime(t, down):
            c.add(t + rng.uniform(0, 1), "inference.scheduler", "INFO",
                  f"dispatched batch_id=b{int(t) % 10 ** 7:07d} size={rng.randint(8, 32)} queue_depth={rng.randint(0, 12)}")
        t += 30
    t = t0 + 70
    while t < end:
        c.add(t + rng.uniform(0, 1), "storage.feature_store", "INFO",
              f"served {rng.randint(40, 120)} feature lookups p95_ms={rng.randint(4, 15)}")
        if int(t - t0) % 600 < 60:
            c.add(t + 3, "storage.feature_store", "INFO", f"feature cache refresh completed keys={rng.randint(20000, 21000)}")
        t += 60
    t = t0 + 90
    while t < end:
        c.add(t + rng.uniform(0, 1), "db.connector", "INFO",
              f"connection pool usage {rng.randint(3, 9)}/20 active, wait queue 0")
        if rng.random() < 0.012:
            c.add(t + 5, "db.connector", "WARN", f"slow query took {rng.randint(1100, 2600)} ms: SELECT features FROM feature_rows")
        t += 120
    t = t0 + 1800
    while t < end:
        if not in_downtime(t, down):
            c.add(t, "inference.model_loader", "INFO", "model resnet50 health check passed on " + GPU_DEVICE)
        t += 1800
    # sparse isolated errors outside incidents
    for at in ("2026-03-09 05:12:44", "2026-03-10 01:33:09", "2026-03-10 20:47:51", "2026-03-11 09:05:17"):
        c.add(ts(at), "api.gateway", "ERROR", f"POST /v1/predict status=500 unhandled exception request_id={c.request_id()}",
              continuation=["Traceback (most recent call last):",
                            '  File "/srv/gateway/handlers.py", line 88, in predict',
                            "    payload = decode_image(request.body)",
                            "ValueError: image payload truncated"])


def generate_service_start(c: EventCollector, s: dict):
    t = ts(s["at"])
    v, b, w, p = s["model_version"], s["batch_size"], s["workers"], s["port"]
    c.add(t, "inference.model_loader", "INFO", "starting inference service (build 2026.03.1)", tag=s["start_id"])
    c.add(t + 1.0, "inference.model_loader", "INFO", "loading configuration from /etc/inference/config.yaml")
    c.add(t + 1.2, "inference.model_loader", "INFO", f"config: batch_size={b} workers={w} port={p} max_queue=256")
    c.add(t + 2.0, "db.connector", "INFO", f"connecting to database {DB_HOST} (pool_size=20)")
    c.add(t + 2.6, "db.connector", "INFO", f"connected to database {DB_HOST} (postgres 16.2)")
    c.add(t + 3.0, "storage.feature_store", "INFO", f"connecting to feature cache {CACHE_HOST}")
    c.add(t + 3.3, "storage.feature_store", "INFO", "feature cache connected")
    c.add(t + 4.0, "inference.model_loader", "INFO", f"loading model resnet50 version {v} from s3://model-registry/resnet50/{v}")
    c.add(t + 4.0 + s["load_seconds"], "inference.model_loader", "INFO",
          f"model resnet50 version {v} loaded in {s['load_seconds']} s (device {GPU_DEVICE})")
    c.add(t + 5.0 + s["load_seconds"], "inference.scheduler", "INFO", f"scheduler started with {w} workers")
    c.add(t + 5.5 + s["load_seconds"], "auth.service", "INFO", "auth service initialized (issuer=idp.internal)")
    c.add(t + 6.0 + s["load_seconds"], "api.gateway", "INFO", f"api gateway listening on port {p}")


def repeat(c, start, end, every, jitter, fn):
    t = start
    while t < end:
        fn(t + c.rng.uniform(0, jitter))
        t += every


def generate_incident(c: EventCollector, inc: dict):
    t = ts(inc["at"])
    rid = c.request_id
    iid = inc["incident_id"]
    if iid.startswith("database_connection_pool"):
        scale = 1.0 if iid.endswith("exhaustion") else 0.5
        c.add(t, "db.connector", "WARN", "connection pool usage 19/20 active, wait queue 4", tag=iid + ":root")
        c.add(t + 20, "db.connector", "WARN", "connection pool usage 20/20 active, wait queue 17")
        repeat(c, t + 45, t + 45 + 360 * scale, 10, 2, lambda x: c.add(x, "db.connector", "ERROR",
               "timeout acquiring connection from pool after 30000 ms"))
        repeat(c, t + 60, t + 60 + 360 * scale, 15, 3, lambda x: c.add(x, "storage.feature_store", "ERROR",
               f"feature lookup failed for request {rid()}: database timeout"))
        repeat(c, t + 70, t + 70 + 300 * scale, 6, 2, lambda x: c.add(x, "api.gateway", "ERROR",
               f"POST /v1/predict status=503 latency_ms=30012 request_id={rid()}", tag=iid + ":symptom"))
        if scale == 1.0:
            c.add(t + 200, "api.gateway", "CRITICAL", "5xx error rate 41.7% over last 60s exceeds threshold 5%")
        c.add(t + 70 + 360 * scale, "db.connector", "INFO", "connection pool usage 6/20 active, wait queue 0", tag=iid + ":resolution")
        c.add(t + 80 + 360 * scale, "api.gateway", "INFO", "5xx error rate back below threshold")
    elif iid == "identity_provider_certificate_expiry":
        c.add(t, "auth.service", "ERROR", "TLS handshake with idp.internal:443 failed: certificate has expired "
              "(notAfter=2026-03-09 22:00:00)", tag=iid + ":root")
        repeat(c, t + 5, t + 485, 8, 2, lambda x: c.add(x, "auth.service", "ERROR",
               "token validation failed: unable to fetch signing keys from idp.internal"))
        repeat(c, t + 10, t + 490, 4, 1.5, lambda x: c.add(x, "api.gateway", "WARN",
               f"POST /v1/predict status=401 unauthorized request_id={rid()}", tag=iid + ":symptom"))
        c.add(t + 240, "api.gateway", "CRITICAL", "4xx error rate 63.0% over last 60s exceeds threshold 20%")
        c.add(t + 500, "auth.service", "INFO", "reloaded trust bundle; certificate for idp.internal valid until 2027-03-09",
              tag=iid + ":resolution")
    elif iid == "feature_cache_connection_refused":
        c.add(t, "storage.feature_store", "ERROR", f"redis connection refused: {CACHE_HOST}", tag=iid + ":root")
        repeat(c, t + 2, t + 362, 12, 2, lambda x: c.add(x, "storage.feature_store", "ERROR",
               f"feature cache unavailable, falling back to cold storage ({CACHE_HOST})"))
        repeat(c, t + 15, t + 375, 10, 2, lambda x: c.add(x, "inference.scheduler", "WARN",
               f"feature fetch for batch_id=b{int(x) % 10 ** 7:07d} exceeded 2000 ms, retrying"))
        repeat(c, t + 30, t + 370, 7, 2, lambda x: c.add(x, "api.gateway", "ERROR",
               f"POST /v1/predict status=504 gateway timeout request_id={rid()}", tag=iid + ":symptom"))
        c.add(t + 380, "storage.feature_store", "INFO", f"redis connection restored: {CACHE_HOST}", tag=iid + ":resolution")
    elif iid == "invalid_configuration_reload":
        c.add(t, "inference.model_loader", "INFO", "configuration change detected in /etc/inference/config.yaml; reloading")
        c.add(t + 1, "inference.model_loader", "ERROR", "invalid configuration value batch_size=-1 (must be a positive integer)",
              tag=iid + ":root")
        c.add(t + 2, "inference.model_loader", "ERROR", "model reload failed; keeping previous model resnet50 version v3.3.0",
              tag=iid + ":symptom")
        repeat(c, t + 5, t + 595, 30, 3, lambda x: c.add(x, "inference.scheduler", "WARN",
               "using stale configuration generation 41; reload pending"))
        c.add(t + 600, "inference.model_loader", "INFO", "configuration reloaded successfully: batch_size=64 workers=8",
              tag=iid + ":resolution")
    elif iid == "gpu_out_of_memory":
        c.add(t, "inference.model_loader", "WARN", f"GPU memory usage 92% (14.7/16.0 GiB) on {GPU_DEVICE}", tag=iid + ":root")
        c.add(t + 40, "inference.model_loader", "WARN", f"GPU memory usage 98% (15.7/16.0 GiB) on {GPU_DEVICE}")
        c.add(t + 65, "inference.model_loader", "ERROR", f"CUDA out of memory: tried to allocate 2.10 GiB ({GPU_DEVICE}; 16.00 GiB total)",
              continuation=["Traceback (most recent call last):",
                            '  File "/srv/inference/runner.py", line 214, in run_batch',
                            "    outputs = model(inputs.to(device))",
                            '  File "/srv/inference/model.py", line 77, in forward',
                            "torch.OutOfMemoryError: CUDA out of memory"])
        c.add(t + 66, "inference.scheduler", "CRITICAL", "worker 3 crashed (exit code 137); restarting", tag=iid + ":symptom")
        for k, off in ((5, 71), (1, 76), (7, 80)):
            c.add(t + off, "inference.scheduler", "ERROR", f"worker {k} crashed (exit code 137)")
        repeat(c, t + 68, t + 68 + 480, 5, 1.5, lambda x: c.add(x, "api.gateway", "ERROR",
               f"POST /v1/predict status=500 internal error request_id={rid()}"))
        c.add(t + 560, "inference.scheduler", "INFO", "all 6 workers healthy after service restart",
              tag=iid + ":resolution")
    elif iid == "disk_full_on_shared_volume":
        c.add(t, "metrics.exporter", "ERROR", "failed to write metrics spool /var/lib/metrics/spool.db: No space left on device",
              tag=iid + ":root")
        repeat(c, t + 30, t + 390, 15, 2, lambda x: c.add(x, "metrics.exporter", "ERROR",
               "failed to write metrics spool /var/lib/metrics/spool.db: No space left on device"))
        repeat(c, t + 90, t + 390, 20, 3, lambda x: c.add(x, "db.connector", "ERROR",
               "write-ahead log append failed: No space left on device (/var/lib/postgresql)", tag=iid + ":symptom"))
        repeat(c, t + 100, t + 380, 8, 2, lambda x: c.add(x, "api.gateway", "ERROR",
               f"POST /v1/predict status=500 internal error request_id={rid()}"))
        c.add(t + 120, "db.connector", "CRITICAL", "database entered read-only mode due to disk pressure")
        c.add(t + 400, "metrics.exporter", "INFO", "disk usage 71% after log rotation; resuming spool writes",
              tag=iid + ":resolution")


def generate(output_dir: str, seed: int) -> dict:
    rng = random.Random(seed)
    c = EventCollector(rng)
    end = (START + timedelta(days=DURATION_DAYS)).timestamp()
    generate_background(c, end)
    for s in SERVICE_STARTS:
        generate_service_start(c, s)
    for inc in INCIDENTS:
        generate_incident(c, inc)
    c.events.sort(key=lambda e: e["epoch"])
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, LOG_FILE_NAME)
    line_no = 0
    truth_events = []
    with open(log_path, "w", encoding="utf-8") as handle:
        for e in c.events:
            line_no += 1
            handle.write(f'{fmt(e["epoch"])} - {e["logger"]} - {e["level"]} - {e["message"]}\n')
            truth_events.append({"line_no": line_no, "epoch": e["epoch"], "ts_str": fmt(e["epoch"]),
                                 "component": e["logger"], "level": e["level"], "message": e["message"], "tag": e["tag"]})
            for extra in e["continuation"]:
                line_no += 1
                handle.write(extra + "\n")
    manifest = {"generator_seed": seed, "log_file": LOG_FILE_NAME, "components": LOGGERS,
                "service_starts": SERVICE_STARTS, "incidents": INCIDENTS, "db_host": DB_HOST, "cache_host": CACHE_HOST,
                "gpu_device": GPU_DEVICE, "total_lines": line_no, "events": truth_events}
    with open(os.path.join(output_dir, MANIFEST_FILE_NAME), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return {"log_path": log_path, "lines": line_no, "events": len(truth_events)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.seed)))
