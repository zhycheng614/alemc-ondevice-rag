"""Composite-score robustness analyses (EXPERIMENT_PLAN §9; R2-minor-2).

Two analyses over the aggregated ALEMC data:

  1. Weight-profile sensitivity: rank the compared systems under each of the
     three weight profiles. This is the table that demonstrates the ranking
     inversion (efficiency-first vs. latency-only / accuracy-first).

  2. Reference-set sensitivity: because min-max normalization is relative to
     the compared set, drop one system at a time, recompute the ranking of the
     remainder under the 'balanced' profile, and report whether the top-ranked
     system is stable. Gives practitioners the guidance R2 asks for.

Reads results/derived/aggregated.csv; writes:
  results/derived/sensitivity_weight.csv
  results/derived/sensitivity_refset.csv

Usage:
  python scoring/sensitivity.py [--carbon medium]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DERIVED_DIR, WEIGHT_PROFILES, COMPOSITE_DELTA
from scoring.composite import compute, load_aggregated, filter_scope


def weight_sensitivity(rows, carbon, delta):
    """One row per system with its rank under each profile."""
    results = compute(rows, carbon=carbon, delta=delta)
    systems = sorted({r["system"] for r in results})
    table = []
    for sysname in systems:
        entry = {"system": sysname}
        for pname in WEIGHT_PROFILES:
            match = next(r for r in results
                         if r["system"] == sysname and r["profile"] == pname)
            entry[f"rank_{pname}"] = match["rank"]
            entry[f"score_{pname}"] = match["S_ALEMC"]
        table.append(entry)
    # detect inversion: does the accuracy/latency winner differ from the
    # efficiency winner?
    return table


def refset_sensitivity(rows, carbon, delta):
    """Drop each system in turn; record top-ranked system of the remainder."""
    full = compute(rows, carbon=carbon, delta=delta)
    balanced_full = sorted([r for r in full if r["profile"] == "balanced"],
                           key=lambda r: r["rank"])
    baseline_top = balanced_full[0]["system"] if balanced_full else None

    out = [dict(dropped="(none / full set)", top_balanced=baseline_top,
                ranking=" > ".join(r["system"] for r in balanced_full))]
    if len(rows) <= 2:
        return out, baseline_top, True

    stable = True
    for i in range(len(rows)):
        subset = rows[:i] + rows[i + 1:]
        sub = compute(subset, carbon=carbon, delta=delta)
        bal = sorted([r for r in sub if r["profile"] == "balanced"],
                     key=lambda r: r["rank"])
        top = bal[0]["system"] if bal else None
        # top is stable if, ignoring the dropped system, the baseline top
        # remains #1 (or was the one dropped)
        dropped = f"{rows[i]['device']}/{rows[i]['config']}"
        if dropped != baseline_top and top != baseline_top:
            stable = False
        out.append(dict(dropped=dropped, top_balanced=top,
                        ranking=" > ".join(r["system"] for r in bal)))
    return out, baseline_top, stable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carbon", default="medium",
                    choices=["low", "medium", "high"])
    ap.add_argument("--delta", type=float, default=COMPOSITE_DELTA)
    ap.add_argument("--scope", default="crosstier",
                    choices=["crosstier", "config", "all"])
    args = ap.parse_args()

    rows = filter_scope(load_aggregated(), args.scope)
    if len(rows) < 2:
        sys.exit(f"[sensitivity] scope '{args.scope}' has <2 systems.")

    # 1. weight sensitivity
    wtab = weight_sensitivity(rows, args.carbon, args.delta)
    wpath = DERIVED_DIR / f"sensitivity_weight_{args.scope}.csv"
    with open(wpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(wtab[0].keys()))
        w.writeheader(); w.writerows(wtab)
    print(f"[sensitivity] weight-profile ranks -> {wpath}")
    for e in wtab:
        ranks = " ".join(f"{p}=#{e[f'rank_{p}']}" for p in WEIGHT_PROFILES)
        print(f"  {e['system']:24s} {ranks}")

    # 2. reference-set sensitivity
    rtab, top, stable = refset_sensitivity(rows, args.carbon, args.delta)
    rpath = DERIVED_DIR / f"sensitivity_refset_{args.scope}.csv"
    with open(rpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rtab[0].keys()))
        w.writeheader(); w.writerows(rtab)
    print(f"[sensitivity] reference-set (drop-one) -> {rpath}")
    print(f"  balanced top = {top}; top-rank stable under drop-one = {stable}")


if __name__ == "__main__":
    main()
