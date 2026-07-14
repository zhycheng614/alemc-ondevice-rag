"""Energy sampler for the RTX 5070 Ti (desktop) — GPU-package power via NVML.

Reports GPU-package power (nvmlDeviceGetPowerUsage), integrated over each
query window. State explicitly in the paper that this is GPU-package power,
not whole-system (a wall meter / RAPL would be needed for that).

EXPERIMENT_PLAN.md §7.3.

Prefers pynvml; falls back to shelling out to `nvidia-smi`.

Usage:
  python samplers/sampler_nvml.py --out results/raw/energy_desktop.csv
  python samplers/sampler_nvml.py --out results/raw/idle_desktop.csv --duration 60

Output CSV columns: t_epoch, power_w  (optionally cpu_power_w if RAPL present)
Feed to samplers/energy_integrate.py alongside the run's per-query CSV.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


def _nvml_reader():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)

        def read():
            return pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
        return read
    except Exception:
        return None


def _smi_reader():
    def read():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()
            return float(out[0]) if out else None
        except Exception:
            return None
    # verify it works once
    return read if read() is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.1)  # ~10 Hz
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()

    read = _nvml_reader() or _smi_reader()
    if read is None:
        print("[sampler_nvml] ERROR: neither pynvml nor nvidia-smi available.",
              file=sys.stderr)
        sys.exit(2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sampler_nvml] sampling every {args.interval}s -> {out}")

    t_end = (time.time() + args.duration) if args.duration else None
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_epoch", "power_w"])
        try:
            while True:
                t = time.time()
                p = read()
                if p is not None:
                    w.writerow([f"{t:.3f}", f"{p:.2f}"])
                    fh.flush()
                    n += 1
                if t_end and time.time() >= t_end:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    print(f"[sampler_nvml] wrote {n} samples -> {out}")


if __name__ == "__main__":
    main()
