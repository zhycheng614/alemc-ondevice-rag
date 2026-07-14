"""No-root energy sampler for an Android phone, driven from the host via adb.

The Snapdragon 8 Elite reference phone (ASUS AI2501 / SM8750) does not grant
shell access to /sys/class/power_supply/battery/current_now (permission
denied) and has no Termux. The accessible no-root signal is the fuel-gauge
**charge counter** exposed by `dumpsys battery` ("Charge counter", in uAh) plus
"voltage" (mV). We integrate energy as:

    E_run [J] = -Delta(charge_uAh) * 1e-6 [Ah] * 3600 [s/h] * V_mean [V]

i.e. charge delta (Ah) x 3600 x mean pack voltage (V). Discharge makes the
counter DECREASE, so -Delta is positive energy consumed. Per-query resolution
is coarse (counter granularity ~1 mAh), so this yields an accurate MEAN energy
per query over the whole run rather than a reliable per-query value; that mean
is the ALEMC energy dimension we report for the phone tier.

The phone MUST be UNPLUGGED (AC/USB powered:false) or the counter rises.

Usage (host, phone on adb):
  # idle baseline (phone idle, screen on-min, unplugged), 120s:
  python samplers/sampler_adb.py --out results/raw/idle_phone.csv --duration 120
  # during the run:
  python samplers/sampler_adb.py --out results/raw/energy_phone.csv

Output CSV columns: t_epoch, charge_uah, voltage_mv
Post-process with energy_integrate.py (adb format is auto-detected there).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MSYS_NO_PATHCONV", "1")


def read_battery():
    """Return (charge_uah, voltage_mv, plugged) from dumpsys battery."""
    try:
        out = subprocess.run(["adb", "shell", "dumpsys", "battery"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None, None, None
    charge = volt = None
    ac = usb = wl = False
    for line in out.splitlines():
        s = line.strip()
        m = re.match(r"Charge counter:\s*(\d+)", s)
        if m:
            charge = int(m.group(1))
        m = re.match(r"voltage:\s*(\d+)", s)
        if m:
            volt = int(m.group(1))
        if s.startswith("AC powered:"):
            ac = "true" in s
        if s.startswith("USB powered:"):
            usb = "true" in s
        if s.startswith("Wireless powered:"):
            wl = "true" in s
    return charge, volt, (ac or usb or wl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()

    charge, volt, plugged = read_battery()
    if charge is None:
        print("[sampler_adb] ERROR: could not read dumpsys battery Charge "
              "counter. Is the phone connected via adb?", file=sys.stderr)
        sys.exit(2)
    if plugged:
        print("[sampler_adb] WARNING: phone is PLUGGED IN — the charge counter "
              "will rise, not fall. Unplug to measure discharge energy.",
              file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sampler_adb] sampling every {args.interval}s -> {out} "
          f"(start charge={charge} uAh, V={volt} mV)")

    t_end = (time.time() + args.duration) if args.duration else None
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["t_epoch", "charge_uah", "voltage_mv"])
        try:
            while True:
                t = time.time()
                charge, volt, _ = read_battery()
                if charge is not None:
                    w.writerow([f"{t:.3f}", charge, volt if volt else ""])
                    fh.flush()
                    n += 1
                if t_end and time.time() >= t_end:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    print(f"[sampler_adb] wrote {n} samples -> {out}")


if __name__ == "__main__":
    main()
