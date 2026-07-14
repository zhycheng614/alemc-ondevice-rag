"""Energy sampler for the Snapdragon X Elite (Windows-on-ARM) laptop.

Samples instantaneous battery discharge power via the WMI BatteryStatus class
(root\\wmi). `DischargeRate` is instantaneous power in mW; `RemainingCapacity`
is in mWh and is used as a whole-run cross-check.

EXPERIMENT_PLAN.md §7.2. Run UNPLUGGED (on battery) — DischargeRate is 0 while
charging. Screen dimmed, airplane mode, background apps closed.

Usage:
  # 60s idle baseline (do this once, device idle, before the run):
  python samplers/sampler_windows.py --out results/raw/idle_laptop.csv --duration 60
  # during the pipeline run (Ctrl-C to stop, or --duration):
  python samplers/sampler_windows.py --out results/raw/energy_laptop.csv

Output CSV columns: t_epoch, discharge_mw, remaining_mwh
Feed to samplers/energy_integrate.py alongside the run's per-query CSV.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


PS_QUERY = (
    "$ErrorActionPreference='SilentlyContinue';"
    "$b = Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus;"
    "if ($b) { '{0} {1}' -f $b.DischargeRate, $b.RemainingCapacity }"
)


def read_battery_powershell():
    """Return (discharge_mw, remaining_mwh) or (None, None)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", PS_QUERY],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not out:
            return None, None
        parts = out.split()
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.5,
                    help="sampling interval seconds (WMI updates ~1 Hz)")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds; default: run until Ctrl-C")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Sanity check on first read.
    mw, mwh = read_battery_powershell()
    if mw is None:
        print("[sampler_windows] ERROR: could not read BatteryStatus WMI class. "
              "Is this a battery device? Are you unplugged?", file=sys.stderr)
        sys.exit(2)
    if mw == 0:
        print("[sampler_windows] WARNING: DischargeRate=0 — likely PLUGGED IN. "
              "Unplug to measure discharge power.", file=sys.stderr)

    print(f"[sampler_windows] sampling every {args.interval}s -> {out}")
    t_end = (time.time() + args.duration) if args.duration else None
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_epoch", "discharge_mw", "remaining_mwh"])
        try:
            while True:
                t = time.time()
                mw, mwh = read_battery_powershell()
                if mw is not None:
                    w.writerow([f"{t:.3f}", mw, mwh])
                    fh.flush()
                    n += 1
                if t_end and time.time() >= t_end:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    print(f"[sampler_windows] wrote {n} samples -> {out}")


if __name__ == "__main__":
    main()
