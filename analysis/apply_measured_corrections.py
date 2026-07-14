"""Apply clean, directly-measured memory and energy figures to the aggregated
per-tier ALEMC rows, replacing values that the automated per-query capture
mismeasured for documented reasons.

Why this step exists (all corrections are toward MORE honest numbers, sourced
from separate clean measurements recorded in results/raw/*.json):

  MEMORY
    - laptop: per-query Windows WorkingSet64 drifted upward over ~800 queries
      (mmap'd GGUF pages + KV growth). Replaced with clean steady-state
      footprint (server WS after 8-query warm-up + pipeline RSS) from
      memory_footprint_laptop.json.
    - phone: generation runs on-device (llama-cli) but the pipeline process
      (host) only sees its own ~300 MB RSS. Replaced with the on-device
      llama-cli peak RSS measured via /proc (2339 MB).
    - desktop: the CUDA llama-server's host working set grows continuously
      over the full 200-query run (~2.0 GB freshly loaded -> ~10.3 GB by the
      end) rather than plateauing after a short warm-up like the CPU tiers;
      reproduced independently on a second fresh server. This is a real,
      measured characteristic of sustained GPU-backed serving (CUDA host
      allocator/buffer caching), not an accounting artifact, so we replace the
      naive per-query MEAN (which understates it, ~6.3 GB, and depends on
      query order) with the directly observed PEAK over the full run, from
      memory_footprint_desktop.json.

  ENERGY
    - phone: the adb charge-counter is too coarse for per-query windows and the
      per-query loop reloads the model each call (I/O energy). Replaced with
      sustained-generation whole-device power (16.6 W, measured) x mean measured
      inference latency, from power_measurements.json. Idle-attributable net
      also computed at 8.8 W.
    - laptop: gross per-query integration is sound; left as measured.

Everything corrected here is a real measurement; the paper states the method.
Run AFTER aggregate.py and BEFORE composite.py / make_*.py.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DERIVED_DIR, RAW_DIR, GRID_INTENSITIES, joules_to_gco2e

AGG = DERIVED_DIR / "aggregated.csv"


def load_json(name):
    p = RAW_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    if not AGG.exists():
        sys.exit("[correct] run aggregate.py first")
    rows = list(csv.DictReader(open(AGG, encoding="utf-8")))

    mem = load_json("memory_footprint_laptop.json")
    desktop_mem = load_json("memory_footprint_desktop.json")
    pw = load_json("power_measurements.json")
    laptop_mem = (mem.get("total_mb") or {})
    desktop_peak_mem = (desktop_mem.get("peak_mb") or {})

    for r in rows:
        dev, cfg = r["device"], r["config"]

        # --- laptop memory: clean steady-state footprint ---
        if dev == "laptop" and cfg in laptop_mem:
            r["peak_rss_mb_mean"] = laptop_mem[cfg]
            r["peak_rss_mb_std"] = 0.0

        # --- phone: clean memory + energy ---
        if dev == "phone":
            r["peak_rss_mb_mean"] = 2339      # on-device llama-cli peak RSS (MB)
            r["peak_rss_mb_std"] = 0.0
            # energy = sustained-generation device power x measured inference e2e
            p_active = (pw.get("phone") or {}).get("sustained_generation_w", 16.6)
            e2e_s = float(r["e2e_ms_mean"]) / 1000.0
            r["energy_j_mean"] = round(p_active * e2e_s, 4)
            # net (inference-attributable) at 8.8 W
            p_net = (pw.get("phone") or {}).get("inference_attributable_w", 8.8)
            r["energy_net_j_mean"] = round(p_net * e2e_s, 4)
            # recompute carbon from corrected energy
            for name, gamma in GRID_INTENSITIES.items():
                r[f"carbon_{name}_mean"] = round(
                    joules_to_gco2e(float(r["energy_j_mean"]), gamma), 8)

        # --- desktop memory: observed peak over the full run (sustained CUDA
        # server growth, not a short-warmup steady state or a naive mean) ---
        if dev == "desktop" and cfg in desktop_peak_mem:
            r["peak_rss_mb_mean"] = desktop_peak_mem[cfg]
            r["peak_rss_mb_std"] = 0.0

    with open(AGG, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[correct] applied measured memory/energy corrections -> {AGG}")
    for r in rows:
        print(f"  {r['device']:8s}/{r['config']:14s} acc={r['accuracy_mean']} "
              f"e2e={r['e2e_ms_mean']}ms E={r['energy_j_mean']}J "
              f"M={r['peak_rss_mb_mean']}MB")


if __name__ == "__main__":
    main()
