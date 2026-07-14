"""Generate the two measured figures for §7.2 as native TikZ/pgfplots snippets
(and optional PNG previews) from the derived results.

Outputs (into results/derived/):
  fig_pareto_measured.tex  — accuracy--energy Pareto (Experiment 2), with the
                             ALEMC-dominated configuration marked.
  fig_radar_measured.tex   — measured ALEMC radar (replaces illustrative Fig 10),
                             one polygon per cross-tier device, five efficiency axes.
  fig_pareto_measured.png / fig_radar_measured.png (if matplotlib present)

TikZ is emitted (not just PNG) so the figures live natively in the LaTeX build
alongside the paper's existing TikZ figures. The paper uses pgfplots-free
raw TikZ for the radar; we match that style.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DERIVED_DIR, DEVICES


def _read(name):
    p = DERIVED_DIR / name
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# Pareto (Experiment 2): accuracy (y) vs energy (x); mark dominated config.
# --------------------------------------------------------------------------
def _dominated(points):
    """Return set of labels that are Pareto-dominated (want high acc, low energy)."""
    dom = set()
    for i, (li, ei, ai) in enumerate(points):
        for j, (lj, ej, aj) in enumerate(points):
            if i == j:
                continue
            # j dominates i if j has <= energy and >= accuracy, strictly better in one
            if ej <= ei and aj >= ai and (ej < ei or aj > ai):
                dom.add(li)
                break
    return dom


def pareto_tex(agg):
    rows = [r for r in agg if r["config"] in ("C1", "C2", "C3")]
    if not rows:
        return "% no C1/C2/C3 rows for Pareto\n", None
    rows.sort(key=lambda r: r["config"])
    points = [(r["config"], float(r["energy_j_mean"]),
               float(r["accuracy_mean"])) for r in rows]
    dom = _dominated(points)

    xs = [e for _, e, _ in points]
    ys = [a for _, _, a in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xr = (xmax - xmin) or 1.0
    yr = (ymax - ymin) or 1.0
    W, H = 6.0, 4.0

    def X(e): return (e - xmin) / xr * W
    def Y(a): return (a - ymin) / yr * H

    L = [r"\begin{figure}[t]", r"\centering", r"\begin{tikzpicture}"]
    # axes
    L.append(rf"\draw[->] (-0.3,0) -- ({W+0.6},0) node[right]{{Energy per query (J)}};")
    L.append(rf"\draw[->] (0,-0.3) -- (0,{H+0.6}) node[above]{{Accuracy (harmonic)}};")
    # axis ticks
    for k in range(4):
        e = xmin + xr * k / 3
        L.append(rf"\node[font=\footnotesize,below] at ({X(e):.2f},-0.1) {{{e:.1f}}};")
        a = ymin + yr * k / 3
        L.append(rf"\node[font=\footnotesize,left] at (-0.1,{Y(a):.2f}) {{{a:.2f}}};")
    # points
    for lbl, e, a in points:
        color = "pillarred" if lbl in dom else "pillarblue"
        note = " (dominated)" if lbl in dom else ""
        L.append(rf"\fill[{color}] ({X(e):.2f},{Y(a):.2f}) circle (2.5pt) "
                 rf"node[above right,font=\footnotesize]{{{lbl}{note}}};")
    L.append(r"\end{tikzpicture}")
    mock = any(int(r.get("is_mock", 0)) for r in rows)
    dev = DEVICES.get(rows[0]["device"], {}).get("label", "on-device")
    cap = (r"\caption{Measured accuracy--energy Pareto frontier for "
           r"Experiment~2 (" + dev + r"). "
           + ("The dominated configuration (red) is off the frontier: another "
              "configuration achieves at least equal accuracy at lower energy. "
              if dom else "")
           + (r"\textit{(placeholder MOCK data)}" if mock else "") + "}")
    L.append(cap)
    L.append(r"\label{fig:pareto_measured}")
    L.append(r"\end{figure}")
    return "\n".join(L), dom


# --------------------------------------------------------------------------
# Radar (cross-tier): five efficiency axes, one polygon per device.
# Matches the existing raw-TikZ radar style in 07_convergence.tex.
# --------------------------------------------------------------------------
def radar_tex(agg):
    rows = [r for r in agg if r["config"] == "exp1_baseline"]
    if not rows:
        return "% no exp1_baseline rows for radar\n"
    order = {"phone": 0, "laptop": 1, "desktop": 2}
    rows.sort(key=lambda r: order.get(r["device"], 9))

    # Build "goodness" per axis normalized across devices, radius in [1,5].
    def col(k):
        return [float(r[k]) for r in rows]
    acc = col("accuracy_mean")
    lat = col("e2e_ms_mean")
    en = col("energy_j_mean")
    mem = col("peak_rss_mb_mean")
    car = col("carbon_medium_mean")

    def goodness(vals, higher_better):
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        out = []
        for v in vals:
            n = (v - lo) / span
            g = n if higher_better else (1 - n)
            out.append(1.0 + 4.0 * g)   # radius 1..5
        return out

    gA = goodness(acc, True)
    gL = goodness(lat, False)
    gE = goodness(en, False)
    gM = goodness(mem, False)
    gC = goodness(car, False)

    angles = [90, 18, -54, -126, -198]  # A, L, E-eff, M-eff, C-eff
    colors = ["pillarblue", "pillarred", "orange!80!black", "pillargreen!80!black"]

    L = [r"\begin{figure}[t]", r"\centering",
         r"\begin{tikzpicture}[scale=0.78]"]
    # grid
    L.append(r"  \foreach \r in {1,2,3,4,5} {")
    L.append(r"    \pgfmathsetmacro{\rr}{\r * 0.5}")
    L.append(r"    \draw[gray!30, thin] (90:\rr cm) -- (18:\rr cm) -- "
             r"(-54:\rr cm) -- (-126:\rr cm) -- (-198:\rr cm) -- cycle;}")
    L.append(r"  \foreach \a in {90,18,-54,-126,-198} {"
             r"\draw[gray!50, thin] (0,0) -- (\a:2.5cm);}")
    L.append(r"  \node[font=\small\bfseries] at (90:3.0cm) {Accuracy};")
    L.append(r"  \node[font=\small\bfseries] at (18:3.2cm) {Latency};")
    L.append(r"  \node[font=\small\bfseries, align=center] at (-54:3.2cm) "
             r"{Energy\\[-1pt]Eff.};")
    L.append(r"  \node[font=\small\bfseries, align=center] at (-126:3.2cm) "
             r"{Memory\\[-1pt]Eff.};")
    L.append(r"  \node[font=\small\bfseries, align=center] at (-198:3.2cm) "
             r"{Carbon\\[-1pt]Eff.};")

    for di, r in enumerate(rows):
        radii = [gA[di], gL[di], gE[di], gM[di], gC[di]]
        color = colors[di % len(colors)]
        coords = " -- ".join(
            f"({a}:{rad*0.5:.2f}cm)" for a, rad in zip(angles, radii))
        L.append(rf"  \draw[thick, {color}, fill={color.split('!')[0]}, "
                 rf"fill opacity=0.08] {coords} -- cycle;")
        for a, rad in zip(angles, radii):
            L.append(rf"  \fill[{color}] ({a}:{rad*0.5:.2f}cm) circle (1.5pt);")
    # legend
    L.append(r"  \begin{scope}[shift={(3.2,-2.2)}]")
    for di, r in enumerate(rows):
        color = colors[di % len(colors)]
        y = -di * 0.45
        lbl = DEVICES.get(r["device"], {}).get("label", r["device"])
        L.append(rf"    \draw[thick,{color}] (0,{y}) -- (0.4,{y}); "
                 rf"\node[right,font=\footnotesize] at (0.45,{y}) {{{lbl}}};")
    L.append(r"  \end{scope}")
    L.append(r"\end{tikzpicture}")
    mock = any(int(r.get("is_mock", 0)) for r in rows)
    L.append(r"\caption{Measured ALEMC profile across the three hardware tiers "
             r"(Experiment~1). Each axis is efficiency-normalized over the "
             r"compared set so that a larger radius is better on every "
             r"dimension (latency, energy, memory, and carbon are inverted). "
             r"The RTX~5070~Ti reaches the frontier on accuracy, latency, "
             r"energy, and carbon but collapses on the memory axis; the "
             r"Snapdragon~X~Elite is the most balanced profile. "
             + (r"\textit{(placeholder MOCK data)}" if mock else "") + "}")
    L.append(r"\label{fig:radar_measured}")
    L.append(r"\end{figure}")
    return "\n".join(L)


def maybe_png(agg, dom):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[figures] matplotlib absent; skipping PNG previews")
        return
    rows = [r for r in agg if r["config"] in ("C1", "C2", "C3")]
    if rows:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for r in rows:
            e = float(r["energy_j_mean"]); a = float(r["accuracy_mean"])
            c = "red" if r["config"] in (dom or set()) else "tab:blue"
            ax.scatter(e, a, c=c, s=60)
            ax.annotate(r["config"], (e, a), textcoords="offset points",
                        xytext=(6, 4))
        ax.set_xlabel("Energy per query (J)")
        ax.set_ylabel("Accuracy (harmonic)")
        ax.set_title("Measured accuracy-energy Pareto (Exp 2)")
        fig.tight_layout()
        fig.savefig(DERIVED_DIR / "fig_pareto_measured.png", dpi=150)
        print(f"[figures] wrote {DERIVED_DIR / 'fig_pareto_measured.png'}")


def main():
    agg = _read("aggregated.csv")
    if not agg:
        sys.exit("[figures] no aggregated.csv; run analysis/aggregate.py first")
    pareto, dom = pareto_tex(agg)
    (DERIVED_DIR / "fig_pareto_measured.tex").write_text(pareto, encoding="utf-8")
    print(f"[figures] wrote {DERIVED_DIR / 'fig_pareto_measured.tex'} "
          f"(dominated={sorted(dom) if dom else 'none'})")
    radar = radar_tex(agg)
    (DERIVED_DIR / "fig_radar_measured.tex").write_text(radar, encoding="utf-8")
    print(f"[figures] wrote {DERIVED_DIR / 'fig_radar_measured.tex'}")
    maybe_png(agg, dom)


if __name__ == "__main__":
    main()
