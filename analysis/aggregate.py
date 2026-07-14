"""Aggregate scored per-query CSVs into per-(device, config) ALEMC summaries.

Reads every scored run CSV in results/raw/, drops warm-up queries, and
computes mean +/- std for each of the five ALEMC dimensions plus the carbon
values at the three reference grid intensities.

Output: results/derived/aggregated.csv, one row per (device, config), with:
  device, config, model, n, is_mock,
  accuracy_mean/std, f1_mean, em_mean, faithfulness_mean,
  ttft_ms_mean/std, e2e_ms_mean/std,
  energy_j_mean/std (idle-subtracted, falls back to gross),
  peak_rss_mb_mean/std,
  carbon_low_mean, carbon_medium_mean, carbon_high_mean

This is the single table every downstream analysis (composite, sensitivity,
figures, LaTeX tables) consumes.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, DERIVED_DIR, GRID_INTENSITIES, joules_to_gco2e


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _f(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _reject_stalls(rows):
    """Drop stall outliers where wall-clock e2e is dominated by an OS hang
    rather than inference. We use a robust upper fence on e2e latency:
    median + 8*MAD (very permissive — only removes true multi-second/minute
    hangs, not normal decode variance). Returns (kept_rows, n_dropped).
    """
    e2e = sorted(v for v in (_f(r, "e2e_ms") for r in rows) if v is not None)
    if len(e2e) < 5:
        return rows, 0
    med = e2e[len(e2e) // 2]
    mad = sorted(abs(v - med) for v in e2e)[len(e2e) // 2] or 1.0
    fence = med + 8.0 * 1.4826 * mad
    kept = [r for r in rows if (_f(r, "e2e_ms") or 0) <= fence]
    return kept, len(rows) - len(kept)


def aggregate_file(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        all_rows = [r for r in csv.DictReader(fh) if int(r.get("warmup", 0)) == 0]
    if not all_rows:
        return None
    rows, n_stall = _reject_stalls(all_rows)
    if n_stall:
        print(f"[aggregate] {path.name}: dropped {n_stall} stall outlier(s) "
              f"(OS hang, not inference)")

    # Energy: report BOTH whole-device gross (directly measured, robust) and
    # inference-attributable idle-subtracted. We use gross as the primary
    # energy dimension because idle-subtraction on a power-efficient SoC leaves
    # many near-idle windows at ~0 (high relative noise); gross is the
    # cross-device-comparable measured quantity. Rows with blank energy
    # (device plugged/charging during the window) are excluded.
    def gross(r):
        return _f(r, "energy_gross_j")

    def net(r):
        return _f(r, "energy_idle_subtracted_j")

    acc = [_f(r, "accuracy_harmonic") for r in rows]
    f1 = [_f(r, "f1") for r in rows]
    em = [_f(r, "em") for r in rows]
    faith = [_f(r, "faithfulness") for r in rows]
    ttft = [_f(r, "ttft_ms") for r in rows]
    e2e = [_f(r, "e2e_ms") for r in rows]
    en = [gross(r) for r in rows if gross(r) is not None]
    en_net = [net(r) for r in rows if net(r) is not None]
    rss = [_f(r, "peak_rss_mb") for r in rows]

    acc_m, acc_s = _mean_std(acc)
    e2e_m, e2e_s = _mean_std(e2e)
    ttft_m, ttft_s = _mean_std(ttft)
    en_m, en_s = _mean_std(en)
    en_net_m, en_net_s = _mean_std(en_net)
    rss_m, rss_s = _mean_std(rss)

    out = dict(
        device=rows[0].get("device", ""),
        config=rows[0].get("config", ""),
        model=rows[0].get("model", ""),
        n=len(rows),
        n_energy=len(en),
        n_stall_dropped=n_stall,
        is_mock=int(any(int(r.get("is_mock", 0)) for r in rows)),
        accuracy_mean=round(acc_m, 4), accuracy_std=round(acc_s, 4),
        f1_mean=round(_mean_std(f1)[0], 4),
        em_mean=round(_mean_std(em)[0], 4),
        faithfulness_mean=round(_mean_std(faith)[0], 4),
        ttft_ms_mean=round(ttft_m, 2), ttft_ms_std=round(ttft_s, 2),
        e2e_ms_mean=round(e2e_m, 2), e2e_ms_std=round(e2e_s, 2),
        energy_j_mean=round(en_m, 4), energy_j_std=round(en_s, 4),
        energy_net_j_mean=round(en_net_m, 4), energy_net_j_std=round(en_net_s, 4),
        peak_rss_mb_mean=round(rss_m, 1), peak_rss_mb_std=round(rss_s, 1),
    )
    for name, gamma in GRID_INTENSITIES.items():
        out[f"carbon_{name}_mean"] = round(joules_to_gco2e(en_m, gamma), 8)
    return out


def main():
    files = [Path(f) for f in glob.glob(str(RAW_DIR / "*.csv"))
             if not Path(f).name.startswith(("idle_", "energy_"))]
    rows = []
    any_mock = False
    for f in sorted(files):
        agg = aggregate_file(f)
        if agg:
            rows.append(agg)
            any_mock = any_mock or agg["is_mock"]
            print(f"[aggregate] {f.name}: acc={agg['accuracy_mean']:.3f} "
                  f"e2e={agg['e2e_ms_mean']:.0f}ms E={agg['energy_j_mean']:.2f}J "
                  f"M={agg['peak_rss_mb_mean']:.0f}MB mock={agg['is_mock']}")
    if not rows:
        print("[aggregate] no scored run CSVs found", file=sys.stderr)
        return
    out = DERIVED_DIR / "aggregated.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {out}  ({len(rows)} device/config rows)")
    if any_mock:
        print("[aggregate] WARNING: some rows are MOCK — not real measurements.")


if __name__ == "__main__":
    main()
