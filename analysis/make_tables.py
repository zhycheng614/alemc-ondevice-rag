"""Generate the LaTeX tables for §7.2 from the derived results.

Produces (into results/derived/):
  table_crosstier.tex   — Experiment 1: 5 ALEMC dims x 3 devices (mean+/-std)
  table_config.tex       — Experiment 2: C1/C2/C3, 5 dims
  table_composite.tex    — composite ranking under 3 weight profiles (inversion)

Tables use booktabs and siunitx-free plain formatting so they drop into the
existing paper without new package deps. Carbon shown at the medium grid
intensity in the main tables (full three-intensity data is in the CSVs).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DERIVED_DIR, DEVICES, CONFIGS, GRID_INTENSITIES


def _read(name):
    path = DERIVED_DIR / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fmt(mean, std=None, prec=2):
    if std is not None and float(std) > 0:
        return f"{float(mean):.{prec}f}$\\pm${float(std):.{prec}f}"
    return f"{float(mean):.{prec}f}"


def _dev_label(d):
    return DEVICES.get(d, {}).get("label", d)


def _sys_label(system):
    """Map a 'device/config' key to a readable, LaTeX-safe label."""
    if "/" in system:
        dev, cfg = system.split("/", 1)
    else:
        dev, cfg = system, ""
    dlabel = _dev_label(dev)
    if cfg in ("", "exp1_baseline"):
        return dlabel
    return f"{dlabel} ({cfg})"


def crosstier_table(agg):
    rows = [r for r in agg if r["config"] == "exp1_baseline"]
    order = {"phone": 0, "laptop": 1, "desktop": 2}
    rows.sort(key=lambda r: order.get(r["device"], 9))
    mock = any(int(r.get("is_mock", 0)) for r in rows)

    n = len(rows)
    tier_word = {1: "one hardware tier", 2: "two hardware tiers",
                 3: "three hardware tiers"}.get(n, f"{n} hardware tiers")
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    cap = (r"\caption{Experiment~1 (cross-tier): the identical RAG pipeline "
           r"(Gemma-2-2B \texttt{Q4\_K\_M}, dense retrieval, $k{=}5$) measured "
           r"across " + tier_word + r". Accuracy is device-independent "
           r"(scored centrally) and held approximately constant by design; "
           r"energy spans a wide range across tiers. Carbon at the medium "
           r"grid intensity (300\,gCO$_2$/kWh).}")
    L.append(cap)
    L.append(r"\label{tab:crosstier}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{lccccc}")
    L.append(r"\toprule")
    L.append(r"Tier / Device & Accuracy & $\mathcal{L}_\text{e2e}$ (ms) & "
             r"Energy (J) & Memory (MB) & Carbon (gCO$_2$e) \\")
    L.append(r"\midrule")
    for r in rows:
        L.append(" & ".join([
            f"{DEVICES.get(r['device'],{}).get('tier','')} / {_dev_label(r['device'])}",
            _fmt(r["accuracy_mean"], r["accuracy_std"], 3),
            _fmt(r["e2e_ms_mean"], r["e2e_ms_std"], 0),
            _fmt(r["energy_j_mean"], r["energy_j_std"], 2),
            _fmt(r["peak_rss_mb_mean"], r["peak_rss_mb_std"], 0),
            f"{float(r['carbon_medium_mean']):.4f}",
        ]) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    if mock:
        L.append(r"\\[2pt]\footnotesize\textit{(placeholder MOCK data — "
                 r"replace with measured runs)}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def config_table(agg):
    rows = [r for r in agg if r["config"] in ("C1", "C2", "C3")]
    rows.sort(key=lambda r: r["config"])
    mock = any(int(r.get("is_mock", 0)) for r in rows)
    dev = _dev_label(rows[0]["device"]) if rows else "on-device"

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{Experiment~2 (config axis, " + dev + r"): three "
             r"RAG configurations mirroring surveyed archetypes. Accuracy rises "
             r"C1$\rightarrow$C3 alongside energy and latency; the "
             r"accuracy--energy Pareto (Fig.~\ref{fig:pareto_measured}) exposes "
             r"the dominated configuration.}")
    L.append(r"\label{tab:config}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{llccc}")
    L.append(r"\toprule")
    L.append(r"Config & Archetype & Accuracy & Energy (J) & $\mathcal{L}_\text{e2e}$ (ms) \\")
    L.append(r"\midrule")
    for r in rows:
        arche = CONFIGS.get(r["config"], {}).get("archetype", "")
        L.append(" & ".join([
            r["config"], arche.split("-style")[0].split("/")[0],
            _fmt(r["accuracy_mean"], r["accuracy_std"], 3),
            _fmt(r["energy_j_mean"], r["energy_j_std"], 2),
            _fmt(r["e2e_ms_mean"], r["e2e_ms_std"], 0),
        ]) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    if mock:
        L.append(r"\\[2pt]\footnotesize\textit{(placeholder MOCK data)}")
    L.append(r"\end{table}")
    return "\n".join(L)


DIM_LABELS = dict(accuracy="accuracy", latency="latency", energy="energy",
                   memory="memory", carbon="carbon")


def _inversion_note(comp, profiles, systems):
    """Describe the real driver of any latency-vs-composite inversion.

    Finds the system with the best raw latency (lowest e2e_ms) and, if it is
    not the top composite rank under a given profile, identifies which
    non-latency dimension has its lowest normalized-goodness score (g_*) —
    i.e. the actual cause of the rank drop, read from the data rather than
    assumed.
    """
    if not comp:
        return ""
    by_system_profile = {(r["profile"], r["system"]): r for r in comp}
    # latency winner = lowest e2e_ms among any one profile's rows (e2e_ms is
    # profile-independent, so the first profile suffices).
    p0 = profiles[0]
    rows_p0 = [r for r in comp if r["profile"] == p0]
    lat_winner = min(rows_p0, key=lambda r: float(r["e2e_ms"]))["system"]

    notes = []
    won_dims = None
    for p in profiles:
        r = by_system_profile.get((p, lat_winner))
        if not r or r["rank"] == "1":
            continue
        dims = ("accuracy", "energy", "memory", "carbon")
        worst_dim = min(dims, key=lambda d: float(r[f"g_{d}"]))
        notes.append((p, worst_dim))
        won_dims = [d for d in dims if d != worst_dim and float(r[f"g_{d}"]) >= 0.999]
    if not notes:
        return ""
    # Report using the dimension that most often drives the drop.
    worst_dim = max(set(d for _, d in notes), key=lambda d: sum(1 for _, dd in notes if dd == d))
    label = _dev_label(lat_winner.split("/", 1)[0])
    won_str = "/".join(DIM_LABELS[d] for d in (won_dims or []))
    won_clause = f" (it wins outright on latency{', ' + won_str if won_str else ''})"
    return (rf" The efficiency-first profile inverts the latency-only ordering: "
            rf"{label}, fastest end-to-end, drops in rank because its "
            rf"{DIM_LABELS[worst_dim]} is the worst in the compared set"
            rf"{won_clause}.")


def composite_table(comp):
    if not comp:
        return "% composite_medium.csv not found\n"
    profiles = []
    for r in comp:
        if r["profile"] not in profiles:
            profiles.append(r["profile"])
    systems = []
    for r in comp:
        if r["system"] not in systems:
            systems.append(r["system"])

    # rank lookup
    rank = {(r["profile"], r["system"]): (r["rank"], r["S_ALEMC"]) for r in comp}

    inv = _inversion_note(comp, profiles, systems)
    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"\centering")
    L.append(r"\caption{ALEMC composite ranking under three weight profiles "
             r"(\S\ref{ssec:alemc}), with $\varepsilon$-smoothing "
             r"($\delta{=}0.05$)." + inv + r"}")
    L.append(r"\label{tab:composite}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{l" + "c" * len(profiles) + "}")
    L.append(r"\toprule")
    L.append("System & " + " & ".join(p.replace("-", "-\\allowbreak ")
                                       for p in profiles) + r" \\")
    L.append(r"\midrule")
    for s in systems:
        cells = []
        for p in profiles:
            rk, sc = rank.get((p, s), ("--", "--"))
            cells.append(f"\\#{rk} ({float(sc):.3f})" if sc != "--" else "--")
        L.append(f"{_sys_label(s)} & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    agg = _read("aggregated.csv")
    comp = _read("composite_crosstier_medium.csv")
    if not agg:
        sys.exit("[tables] no aggregated.csv; run analysis/aggregate.py first")

    outputs = {
        "table_crosstier.tex": crosstier_table(agg),
        "table_config.tex": config_table(agg),
        "table_composite.tex": composite_table(comp),
    }
    for name, tex in outputs.items():
        (DERIVED_DIR / name).write_text(tex, encoding="utf-8")
        print(f"[tables] wrote {DERIVED_DIR / name}")


if __name__ == "__main__":
    main()
