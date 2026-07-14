"""Orchestrate the full X Elite laptop tier run with real energy sampling.

Sequence:
  1. 60 s idle baseline (battery discharge while idle) -> idle_laptop.csv
  2. Start the battery sampler in the background.
  3. Run Exp 1 baseline (200 queries) + Exp 2 configs C1/C2/C3 on the phone-
     equivalent config set (here on the laptop, per the plan's laptop tier we
     only need exp1_baseline; C1/C2/C3 belong to Experiment 2 on the phone but
     we also run them here to exercise the config axis if requested).
  4. Stop the sampler; integrate energy into each run CSV.

Assumes the laptop is UNPLUGGED and llama-server is running
(ALEMC_LLAMA_SERVER set). Configs to run are given on the CLI.

Usage:
  ALEMC_LLAMA_SERVER=http://127.0.0.1:8080 \
  python run_laptop_tier.py --configs exp1_baseline --idle 60
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import RAW_DIR, WARMUP_QUERIES  # noqa: E402
from samplers.sampler_windows import read_battery_powershell  # noqa: E402


def ensure_on_battery():
    mw, _ = read_battery_powershell()
    if mw is None:
        print("[laptop] WARNING: WMI BatteryStatus unavailable; energy will be "
              "empty.", file=sys.stderr)
        return False
    if mw == 0:
        print("[laptop] ERROR: laptop appears PLUGGED IN (DischargeRate=0). "
              "Unplug and retry.", file=sys.stderr)
        return False
    print(f"[laptop] on battery, discharging at {mw/1000:.2f} W")
    return True


def run(cmd, **kw):
    print("[laptop] $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["exp1_baseline"])
    ap.add_argument("--idle", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="laptop")
    args = ap.parse_args()

    py = sys.executable
    if not ensure_on_battery():
        sys.exit(2)

    idle_csv = RAW_DIR / f"idle_{args.device}.csv"
    energy_csv = RAW_DIR / f"energy_{args.device}.csv"

    # 1. idle baseline
    print(f"[laptop] measuring {args.idle}s idle baseline...")
    run([py, str(ROOT / "samplers" / "sampler_windows.py"),
         "--out", str(idle_csv), "--duration", str(args.idle)])

    # 2. start sampler for the whole active run
    print("[laptop] starting energy sampler for the run window...")
    sampler = subprocess.Popen(
        [py, str(ROOT / "samplers" / "sampler_windows.py"),
         "--out", str(energy_csv), "--interval", "0.5"])
    time.sleep(2)

    run_csvs = []
    try:
        # 3. run each config
        for cfg in args.configs:
            out = RAW_DIR / f"{args.device}_{cfg}.csv"
            cmd = [py, str(ROOT / "pipeline" / "run_experiment.py"),
                   "--device", args.device, "--config", cfg, "--out", str(out)]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            run(cmd)
            run_csvs.append(out)
    finally:
        # 4. stop sampler
        print("[laptop] stopping sampler...")
        sampler.terminate()
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()

    # 5. integrate energy into each run CSV
    for out in run_csvs:
        run([py, str(ROOT / "samplers" / "energy_integrate.py"),
             "--run", str(out), "--power", str(energy_csv),
             "--idle", str(idle_csv)])

    print(f"[laptop] done. runs: {[c.name for c in run_csvs]}")
    print(f"[laptop] warmup queries excluded from stats: {WARMUP_QUERIES}")


if __name__ == "__main__":
    main()
