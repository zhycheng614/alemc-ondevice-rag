"""SMOKE TEST ONLY — validates the analysis chain end-to-end on synthetic data.

This is NOT a data generator for the paper. It fabricates device-appropriate
power logs and query timing so that energy_integrate.py, aggregate.py,
composite.py, sensitivity.py, make_tables.py and make_figures.py can be
exercised and the ranking-inversion logic verified BEFORE real model runs.

Every row produced carries is_mock=1. Nothing here may be reported.

It rewrites the mock raw CSVs with (a) spread-out query windows and (b) a
matching synthetic power log per device, then calls the REAL energy_integrate.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR
from samplers.energy_integrate import process_run

# Plausible power / latency / memory / accuracy profiles per device+config.
# Purely illustrative — chosen so the intended inversion is *possible*, then
# verified by the real composite code (not asserted here).
DEVICE_POWER_W = {"phone": 4.5, "laptop": 14.0, "desktop": 240.0}
DEVICE_IDLE_W = {"phone": 1.2, "laptop": 3.5, "desktop": 60.0}
# e2e seconds per query (desktop fastest, phone slowest)
DEVICE_E2E_S = {"phone": 4.0, "laptop": 2.2, "desktop": 0.8}
DEVICE_RSS = {"phone": 2600, "laptop": 3100, "desktop": 5200}
CONFIG_MULT = {  # config axis on phone: C1<C2<C3 in cost and accuracy
    "exp1_baseline": (1.0, 1.0, 0.62),
    "C1": (0.7, 0.8, 0.55), "C2": (1.0, 1.0, 0.66), "C3": (1.9, 1.7, 0.71),
}


def rewrite(run_csv: Path):
    dev = run_csv.stem.split("_")[0]
    cfg = run_csv.stem[len(dev) + 1:]
    pmul, tmul, acc = CONFIG_MULT.get(cfg, (1.0, 1.0, 0.6))
    e2e_s = DEVICE_E2E_S[dev] * tmul
    base_power = DEVICE_POWER_W[dev] * pmul
    idle = DEVICE_IDLE_W[dev]

    with open(run_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())

    t = 1_700_000_000.0  # fixed epoch base (no RNG / Date.now)
    power_log = [(t - 5.0, idle)]  # lead-in idle
    for i, r in enumerate(rows):
        r["t_query_start"] = f"{t:.3f}"
        t_end = t + e2e_s
        r["t_query_end"] = f"{t_end:.3f}"
        r["e2e_ms"] = round(e2e_s * 1000, 2)
        r["ttft_ms"] = round(e2e_s * 1000 * 0.25, 2)
        r["peak_rss_mb"] = DEVICE_RSS[dev]
        # active power samples across the window (flat + small ramp)
        power_log.append((t + 0.01, base_power))
        power_log.append((t_end - 0.01, base_power))
        power_log.append((t_end + 0.05, idle))
        t = t_end + 0.1

    with open(run_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # write synthetic power + idle logs and integrate with the REAL code
    plog = RAW_DIR / f"energy_{run_csv.stem}.csv"
    with open(plog, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["t_epoch", "power_w"])
        for ts, p in power_log:
            w.writerow([f"{ts:.3f}", f"{p:.2f}"])
    ilog = RAW_DIR / f"idle_{run_csv.stem}.csv"
    with open(ilog, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["t_epoch", "power_w"])
        for k in range(20):
            w.writerow([f"{1_700_000_000.0 + k:.3f}", f"{idle:.2f}"])

    process_run(run_csv, plog, ilog)


def main():
    runs = [p for p in RAW_DIR.glob("*.csv")
            if not p.name.startswith(("idle_", "energy_"))]
    for r in sorted(runs):
        rewrite(r)
    print(f"[smoke] rewrote + energy-integrated {len(runs)} synthetic runs")


if __name__ == "__main__":
    main()
