"""Fold per-device power logs into per-query energy (Joules).

For each raw run CSV (with t_query_start / t_query_end epoch columns), read the
matching power sampler log, trapezoidally integrate power over each query
window, and write energy_gross_j. Subtract the idle baseline (mean idle power x
window duration) to write energy_idle_subtracted_j (EXPERIMENT_PLAN §6.3).

Power log formats (auto-detected from header):
  * Windows : t_epoch, discharge_mw, remaining_mwh   (mW -> W = /1000)
  * Android : t_epoch, current_ua, voltage_uv         (P = |uA|*uV/1e12 W)
  * NVML    : t_epoch, power_w                         (already W)

Usage:
  python samplers/energy_integrate.py \
      --run    results/raw/laptop_exp1.csv \
      --power  results/raw/energy_laptop.csv \
      --idle   results/raw/idle_laptop.csv

Or batch mode with a mapping JSON (run_csv -> {power, idle}):
  python samplers/energy_integrate.py --map results/raw/energy_map.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_power_log(path: Path) -> List[Tuple[float, float]]:
    """Return sorted [(t_epoch, power_w), ...] regardless of sampler format."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        cols = [c.strip() for c in header]
        raw = [row for row in reader if row]

    # charge-counter format (adb, no-root): convert cumulative uAh -> power (W)
    # by differencing consecutive samples. P = -dCharge[Ah] * 3600 * V[V] / dt.
    if "charge_uah" in cols:
        pts = []
        for row in raw:
            t = float(row[0])
            q_uah = float(row[1])
            v_mv = float(row[2]) if len(row) > 2 and row[2] else 0.0
            pts.append((t, q_uah, v_mv))
        pts.sort()
        samples = []
        for (t0, q0, v0), (t1, q1, v1) in zip(pts, pts[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            dq_ah = (q1 - q0) * 1e-6          # uAh -> Ah (negative on discharge)
            v = ((v0 + v1) / 2.0) / 1000.0     # mV -> V
            p = -dq_ah * 3600.0 * v / dt       # W (positive when discharging)
            # stamp power at the midpoint of the interval
            samples.append(((t0 + t1) / 2.0, max(0.0, p)))
        samples.sort()
        return samples

    samples = []
    for row in raw:
        t = float(row[0])
        if "discharge_mw" in cols:
            p = float(row[1]) / 1000.0
        elif "current_ua" in cols:
            i_ua = abs(float(row[1]))
            v_uv = float(row[2])
            p = i_ua * v_uv / 1e12
        elif "power_w" in cols:
            p = float(row[1])
        else:
            raise ValueError(f"unrecognized power log header: {cols}")
        samples.append((t, p))
    samples.sort()
    return samples


def integrate_window(samples: List[Tuple[float, float]],
                     t0: float, t1: float) -> float:
    """Trapezoidal integral of power (W) over [t0, t1] -> Joules.

    Interpolates power at the exact window edges; ignores windows with no
    samples (returns 0.0, flagged by caller).
    """
    if t1 <= t0 or not samples:
        return 0.0

    def interp(t):
        # nearest-neighbour clamp for edges outside sample range
        if t <= samples[0][0]:
            return samples[0][1]
        if t >= samples[-1][0]:
            return samples[-1][1]
        lo, hi = 0, len(samples) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if samples[mid][0] < t:
                lo = mid + 1
            else:
                hi = mid
        (ta, pa), (tb, pb) = samples[lo - 1], samples[lo]
        if tb == ta:
            return pb
        frac = (t - ta) / (tb - ta)
        return pa + frac * (pb - pa)

    pts = [(t0, interp(t0))]
    pts += [(t, p) for (t, p) in samples if t0 < t < t1]
    pts.append((t1, interp(t1)))
    energy = 0.0
    for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
        energy += 0.5 * (pa + pb) * (tb - ta)
    return energy


def mean_idle_power(idle_path: Path) -> float:
    samples = load_power_log(idle_path)
    if not samples:
        return 0.0
    return sum(p for _, p in samples) / len(samples)


def process_run(run_csv: Path, power_csv: Path, idle_csv: Path = None):
    samples = load_power_log(power_csv)
    idle_w = mean_idle_power(idle_csv) if idle_csv and Path(idle_csv).exists() else 0.0
    print(f"[energy] {run_csv.name}: {len(samples)} power samples, "
          f"idle={idle_w:.3f} W")

    with open(run_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fields = rows[0].keys() if rows else []

    n_covered = 0
    n_invalid = 0
    for r in rows:
        try:
            t0 = float(r["t_query_start"]); t1 = float(r["t_query_end"])
        except (KeyError, ValueError):
            continue
        dur = max(0.0, t1 - t0)
        e_gross = integrate_window(samples, t0, t1)
        # Validity: if the window's mean power is implausibly low (<1 W), the
        # device was plugged in / charging during that window (DischargeRate=0)
        # and the energy is meaningless. Leave blank so stats exclude it.
        mean_w = (e_gross / dur) if dur > 0 else 0.0
        if mean_w < 1.0:
            r["energy_gross_j"] = ""
            r["energy_idle_subtracted_j"] = ""
            n_invalid += 1
            continue
        e_net = max(0.0, e_gross - idle_w * dur)
        r["energy_gross_j"] = round(e_gross, 4)
        r["energy_idle_subtracted_j"] = round(e_net, 4)
        n_covered += 1

    with open(run_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    print(f"[energy] {run_csv.name}: energy filled for {n_covered}/{len(rows)} "
          f"queries ({n_invalid} excluded: device plugged/charging) -> {run_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--power")
    ap.add_argument("--idle")
    ap.add_argument("--map", help="JSON: {run_csv: {power:..., idle:...}}")
    args = ap.parse_args()

    if args.map:
        mapping = json.loads(Path(args.map).read_text())
        for run_csv, spec in mapping.items():
            process_run(Path(run_csv), Path(spec["power"]),
                        Path(spec["idle"]) if spec.get("idle") else None)
    elif args.run and args.power:
        process_run(Path(args.run), Path(args.power),
                    Path(args.idle) if args.idle else None)
    else:
        ap.error("provide --run and --power, or --map")


if __name__ == "__main__":
    main()
