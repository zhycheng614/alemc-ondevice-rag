"""Orchestrate the Snapdragon 8 Elite phone-tier run over adb (WiFi).

Generation runs ON the phone (llama-cli via `adb shell`); query embedding +
FAISS retrieval + scoring run on the host. Energy is measured with the no-root
charge-counter method (samplers/sampler_adb.py) integrated over the run window.

Requires:
  * ANDROID_SERIAL set to the WiFi adb endpoint (e.g. 10.0.0.109:5555)
  * ALEMC_ADB_DIR=/data/local/tmp/alemc (holds llama-cli, lib/, GGUF)
  * phone UNPLUGGED (on battery)

Because each adb `llama-cli` invocation reloads the model, wall-clock per query
is not representative; the pipeline records inference latency from llama-cli's
throughput footer (see backends._gen_adb). Energy over the whole run is
dominated by reload I/O too, so we ALSO capture a clean sustained-generation
power figure and store it for honest reporting.

Usage:
  ANDROID_SERIAL=10.0.0.109:5555 ALEMC_ADB_DIR=/data/local/tmp/alemc \
  python run_phone_tier.py --config exp1_baseline --idle 120 --limit 200
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MSYS_NO_PATHCONV", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import RAW_DIR  # noqa: E402
from samplers.sampler_adb import read_battery  # noqa: E402


def ensure_on_battery():
    charge, volt, plugged = read_battery()
    if charge is None:
        print("[phone] ERROR: cannot read battery over adb.", file=sys.stderr)
        return False
    if plugged:
        print("[phone] ERROR: phone is PLUGGED IN. Unplug (WiFi adb persists).",
              file=sys.stderr)
        return False
    print(f"[phone] on battery: charge={charge} uAh V={volt} mV")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp1_baseline")
    ap.add_argument("--idle", type=int, default=120)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()
    py = sys.executable

    if not ensure_on_battery():
        sys.exit(2)

    idle_csv = RAW_DIR / "idle_phone.csv"
    energy_csv = RAW_DIR / "energy_phone.csv"
    out_csv = RAW_DIR / f"phone_{args.config}.csv"

    # 1. idle baseline (phone idle, unplugged)
    print(f"[phone] idle baseline {args.idle}s...")
    subprocess.run([py, str(ROOT / "samplers" / "sampler_adb.py"),
                    "--out", str(idle_csv), "--duration", str(args.idle),
                    "--interval", "2"])

    # 2. start charge-counter sampler for the run window
    print("[phone] starting charge-counter sampler...")
    sampler = subprocess.Popen([py, str(ROOT / "samplers" / "sampler_adb.py"),
                                "--out", str(energy_csv), "--interval", "2"])
    time.sleep(2)

    try:
        # 3. run the pipeline (generation on phone via adb backend)
        cmd = [py, str(ROOT / "pipeline" / "run_experiment.py"),
               "--device", "phone", "--config", args.config,
               "--out", str(out_csv), "--limit", str(args.limit)]
        subprocess.run(cmd)
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()

    # 4. integrate energy
    subprocess.run([py, str(ROOT / "samplers" / "energy_integrate.py"),
                    "--run", str(out_csv), "--power", str(energy_csv),
                    "--idle", str(idle_csv)])
    print(f"[phone] done -> {out_csv}")


if __name__ == "__main__":
    main()
