"""ALEMC composite score with epsilon-smoothing (eq:composite; R2-minor-2).

S_ALEMC = A^{w_a} * L^{-w_l} * E^{-w_e} * M^{-w_m} * C^{-w_c}

where each dimension is min-max normalized over the compared set. Accuracy is
"higher is better"; latency/energy/memory/carbon are "lower is better", so we
normalize them as efficiency = 1 - normalized_cost (also higher is better) and
apply POSITIVE exponents to all five normalized-goodness terms. This is
algebraically equivalent to the paper's negative-exponent-on-cost form but is
numerically clean.

Epsilon-smoothing (EXPERIMENT_PLAN §9): every normalized goodness is mapped to
[DELTA, 1] instead of [0, 1], so a worst-in-set value cannot become 0 and drive
the geometric mean (with any positive weight) to 0 / a hard tie at the bottom.
DELTA is reported in the paper.

Reads results/derived/aggregated.csv; writes one ranking CSV per weight profile
plus a combined long-form CSV, into results/derived/.

Usage:
  python scoring/composite.py                 # all profiles
  python scoring/composite.py --carbon medium # pick grid intensity for C dim
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (DERIVED_DIR, WEIGHT_PROFILES, COMPOSITE_DELTA)


def minmax_goodness(values, higher_is_better, delta):
    """Normalize to [delta, 1]; for cost dims invert so higher=better."""
    lo, hi = min(values), max(values)
    span = hi - lo
    out = []
    for v in values:
        if span == 0:
            g = 1.0  # all equal -> neutral top
        else:
            n = (v - lo) / span               # in [0,1], higher raw value
            g = n if higher_is_better else (1.0 - n)
        # map [0,1] -> [delta, 1]
        out.append(delta + (1.0 - delta) * g)
    return out


def load_aggregated():
    path = DERIVED_DIR / "aggregated.csv"
    if not path.exists():
        sys.exit(f"[composite] missing {path}; run analysis/aggregate.py first")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def filter_scope(rows, scope):
    """Restrict to a comparison set so min-max normalization is meaningful.

      * 'crosstier' : the three Experiment-1 tiers (config == exp1_baseline).
                      This is where the ranking-inversion claim lives.
      * 'config'    : the Experiment-2 config axis (C1/C2/C3) on the phone.
      * 'all'       : every system (mixed; use with care — normalization then
                      spans heterogeneous comparison sets).
    """
    if scope == "crosstier":
        return [r for r in rows if r["config"] == "exp1_baseline"]
    if scope == "config":
        return [r for r in rows if r["config"] in ("C1", "C2", "C3")]
    return rows


def compute(rows, carbon="medium", delta=COMPOSITE_DELTA):
    """Return list of dicts: one per system, per profile, with S_ALEMC + rank."""
    label = [f"{r['device']}/{r['config']}" for r in rows]
    A = [float(r["accuracy_mean"]) for r in rows]
    L = [float(r["e2e_ms_mean"]) for r in rows]
    E = [float(r["energy_j_mean"]) for r in rows]
    M = [float(r["peak_rss_mb_mean"]) for r in rows]
    C = [float(r[f"carbon_{carbon}_mean"]) for r in rows]

    gA = minmax_goodness(A, True, delta)
    gL = minmax_goodness(L, False, delta)
    gE = minmax_goodness(E, False, delta)
    gM = minmax_goodness(M, False, delta)
    gC = minmax_goodness(C, False, delta)

    results = []
    for pname, w in WEIGHT_PROFILES.items():
        scored = []
        for i in range(len(rows)):
            s = (gA[i] ** w["a"] * gL[i] ** w["l"] * gE[i] ** w["e"]
                 * gM[i] ** w["m"] * gC[i] ** w["c"])
            scored.append((label[i], s, i))
        scored.sort(key=lambda x: -x[1])
        for rank, (lbl, s, i) in enumerate(scored, 1):
            results.append(dict(
                profile=pname, system=lbl, rank=rank,
                S_ALEMC=round(s, 5),
                g_accuracy=round(gA[i], 4), g_latency=round(gL[i], 4),
                g_energy=round(gE[i], 4), g_memory=round(gM[i], 4),
                g_carbon=round(gC[i], 4),
                accuracy=A[i], e2e_ms=L[i], energy_j=E[i],
                peak_rss_mb=M[i], carbon_gco2e=C[i],
            ))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carbon", default="medium",
                    choices=["low", "medium", "high"])
    ap.add_argument("--delta", type=float, default=COMPOSITE_DELTA)
    ap.add_argument("--scope", default="crosstier",
                    choices=["crosstier", "config", "all"],
                    help="comparison set to normalize over (default: the "
                         "Experiment-1 cross-tier set where the inversion lives)")
    args = ap.parse_args()

    all_rows = load_aggregated()
    if any(int(r.get("is_mock", 0)) for r in all_rows):
        print("[composite] WARNING: aggregated data contains MOCK rows.")
    rows = filter_scope(all_rows, args.scope)
    if len(rows) < 2:
        sys.exit(f"[composite] scope '{args.scope}' has <2 systems; "
                 f"cannot normalize.")
    results = compute(rows, carbon=args.carbon, delta=args.delta)

    out = DERIVED_DIR / f"composite_{args.scope}_{args.carbon}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"[composite] scope={args.scope} delta={args.delta} "
          f"carbon={args.carbon} -> {out}")

    # pretty print the rankings + flag the cross-tier inversion
    lat_winner = min(rows, key=lambda r: float(r["e2e_ms_mean"]))
    lat_winner_lbl = f"{lat_winner['device']}/{lat_winner['config']}"
    for pname in WEIGHT_PROFILES:
        ranked = [r for r in results if r["profile"] == pname]
        order = " > ".join(f"{r['system']}({r['S_ALEMC']:.3f})" for r in ranked)
        print(f"  {pname:16s}: {order}")
    eff = [r for r in results if r["profile"] == "efficiency-first"]
    eff.sort(key=lambda r: r["rank"])
    if eff and eff[0]["system"] != lat_winner_lbl:
        print(f"[composite] INVERSION: latency winner = {lat_winner_lbl}, but "
              f"efficiency-first ALEMC winner = {eff[0]['system']}")


if __name__ == "__main__":
    main()
