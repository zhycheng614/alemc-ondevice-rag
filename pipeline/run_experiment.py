"""Drive one device x config run over the frozen query set; write raw CSV.

Usage (on each device):
  python pipeline/run_experiment.py --device laptop --config exp1_baseline \
         --out results/raw/laptop_exp1.csv

The energy sampler for the device must be running concurrently, writing its
own timestamped power log. energy_integrate.py later joins the two on the
per-query [t_query_start, t_query_end] window.

Writes one CSV row per query with the full data schema (EXPERIMENT_PLAN §8).
The first WARMUP_QUERIES rows are marked warmup=1 and excluded from stats.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (QUERIES_PATH, DEVICES, CONFIGS, GEN_MODEL_NAME,
                    WARMUP_QUERIES, RAW_DIR)
from pipeline.rag_pipeline import RAGPipeline

CSV_FIELDS = [
    "device", "config", "model", "query_id", "dataset", "warmup",
    "ttft_ms", "e2e_ms", "t_retrieval_ms", "t_prefill_ms", "t_decode_ms",
    "n_tokens", "peak_rss_mb",
    "t_query_start", "t_query_end",
    # energy columns filled later by energy_integrate.py:
    "energy_gross_j", "energy_idle_subtracted_j",
    "answer_text", "retrieved_ids", "is_mock",
]


def load_queries(limit=None):
    # ALEMC_QUERIES lets a device run a frozen subset (e.g. the phone tier)
    # while scoring still uses the full gold set.
    import os
    path = os.environ.get("ALEMC_QUERIES", str(QUERIES_PATH))
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=list(DEVICES))
    ap.add_argument("--config", default="exp1_baseline", choices=list(CONFIGS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of queries (for smoke tests)")
    ap.add_argument("--model", default=GEN_MODEL_NAME)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else \
        RAW_DIR / f"{args.device}_{args.config}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.limit)
    pipe = RAGPipeline(args.config)

    print(f"[run] device={args.device} config={args.config} "
          f"queries={len(queries)}")
    print(f"[run] embedder={pipe.embedder.backend} "
          f"generator={pipe.generator.backend} "
          f"retriever_ready={pipe.retriever.ready} MOCK={pipe.is_mock}")
    if pipe.is_mock:
        print("[run] WARNING: running in MOCK mode — results are NOT real "
              "and must not be reported.")

    run_start = time.time()
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for i, q in enumerate(queries):
            rec = pipe.run_query(q["query_id"], q["question"], q["dataset"])
            row = asdict(rec)
            row.update(dict(
                device=args.device, config=args.config, model=args.model,
                warmup=int(i < WARMUP_QUERIES),
                energy_gross_j="", energy_idle_subtracted_j="",
            ))
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(queries)} "
                      f"(e2e={rec.e2e_ms:.0f}ms rss={rec.peak_rss_mb:.0f}MB)")

    dur = time.time() - run_start
    # Write a run manifest with the exact wall-clock window and backends used,
    # so the sampler log can be aligned and provenance is auditable.
    manifest = out_path.with_suffix(".manifest.json")
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump(dict(
            device=args.device, config=args.config, model=args.model,
            n_queries=len(queries), warmup=WARMUP_QUERIES,
            embedder_backend=pipe.embedder.backend,
            generator_backend=pipe.generator.backend,
            is_mock=pipe.is_mock,
            run_start=run_start, run_end=time.time(), duration_s=dur,
        ), fh, indent=2)
    print(f"[run] wrote {out_path}  ({dur:.1f}s)  manifest={manifest.name}")


if __name__ == "__main__":
    main()
