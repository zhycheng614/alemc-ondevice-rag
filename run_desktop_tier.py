"""Orchestrate the RTX 5070 Ti desktop-tier run with NVML energy sampling.

Mirrors run_laptop_tier.py / run_phone_tier.py. Generation runs via a resident
llama-server (CUDA build) or llama.cpp CUDA CLI; embedding + retrieval + scoring
run in the pipeline; energy is GPU-package power from NVML integrated per query.

Prereqs on the desktop (see EXPERIMENT_PLAN.md handoff section):
  * Copy the whole experiments/ tree (code + data/index.faiss + data/queries.jsonl
    + data/passages*.  The 12k index and 200-query set are already built and
    committed, so DO NOT rebuild them — reuse for identical retrieval.)
  * Build llama.cpp with CUDA and start llama-server:
        llama-server -m gemma-2-2b-it-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
    then set ALEMC_LLAMA_SERVER=http://127.0.0.1:8080
  * pip install faiss-cpu fastembed  (matplotlib for figures)
  * pip install pynvml  (or have nvidia-smi on PATH) for the energy sampler

Usage:
  ALEMC_LLAMA_SERVER=http://127.0.0.1:8080 \
  python run_desktop_tier.py --config exp1_baseline --idle 60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import RAW_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp1_baseline")
    ap.add_argument("--idle", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    py = sys.executable

    idle_csv = RAW_DIR / "idle_desktop.csv"
    energy_csv = RAW_DIR / "energy_desktop.csv"
    out_csv = RAW_DIR / f"desktop_{args.config}.csv"

    # 1. GPU idle baseline
    print(f"[desktop] NVML idle baseline {args.idle}s...")
    subprocess.run([py, str(ROOT / "samplers" / "sampler_nvml.py"),
                    "--out", str(idle_csv), "--duration", str(args.idle),
                    "--interval", "0.1"])

    # 2. start NVML sampler for the run window
    print("[desktop] starting NVML power sampler...")
    sampler = subprocess.Popen([py, str(ROOT / "samplers" / "sampler_nvml.py"),
                                "--out", str(energy_csv), "--interval", "0.1"])
    time.sleep(2)

    try:
        cmd = [py, str(ROOT / "pipeline" / "run_experiment.py"),
               "--device", "desktop", "--config", args.config,
               "--out", str(out_csv)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        subprocess.run(cmd)
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()

    # 3. integrate GPU-package energy
    subprocess.run([py, str(ROOT / "samplers" / "energy_integrate.py"),
                    "--run", str(out_csv), "--power", str(energy_csv),
                    "--idle", str(idle_csv)])
    print(f"[desktop] done -> {out_csv}")
    print("[desktop] NOTE: energy is GPU-PACKAGE power (NVML), not whole-system. "
          "State this in the paper.")


if __name__ == "__main__":
    main()
