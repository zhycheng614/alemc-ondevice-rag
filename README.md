# ALEMC Empirical Study — Reproducibility Bundle

This repository is the reproducibility bundle for the empirical validation of
the **ALEMC** framework (Accuracy, Latency, Energy, Memory, Carbon) reported in
§7 of the accompanying paper:

> **Toward Sustainable On-Device Intelligence: A Survey on Energy-Efficient RAG
> Systems with Small Language Models.** Zhiyuan Cheng, Longying Lai, Yue Liu, Yu Sun.

It is a deliberately **scoped proof-of-concept** instantiation of the full
benchmark protocol, not a definitive benchmark. The goal is to show ALEMC works
end-to-end and *reorders systems* under different weight profiles.

> ### 📌 Authoritative inputs (read before reproducing)
>
> The **frozen data files shipped in `data/` are the authoritative inputs behind
> every number in the paper** and should be used as-is to reproduce the results:
> `data/queries.jsonl` (200 queries: **100 SQuAD** factoid + **100 HotpotQA**
> multi-hop) and `data/corpus.jsonl` (**12,000** Wikipedia-derived passages),
> together with the prebuilt `data/index.faiss`.
>
> **Provenance note:** the `data/prepare_*.py` scripts and `config.py`
> (`CORPUS_SIZE = 20_000`, `N_QUERIES_NQ`) document an *earlier* construction that
> used Natural Questions for the factoid split and a 20k corpus target. The study
> later substituted **SQuAD** for Natural Questions — so that every factoid
> question's gold passage is present in the scoped corpus (paper §7.4) — and
> scoped the corpus to **12k**. Re-running the prep scripts therefore re-samples
> the source datasets and will **not** byte-reproduce the shipped frozen files;
> use the committed `data/` files to reproduce the paper.

## What this produces

* **Experiment 1 (cross-tier):** the identical RAG pipeline
  (Gemma-2-2B `Q4_K_M`, dense retrieval, `k=5`) run on three hardware tiers
  (Snapdragon 8 Elite phone / Snapdragon X Elite laptop / RTX 5070 Ti desktop),
  demonstrating a **ranking inversion**: the desktop wins on latency but loses
  under the efficiency-first ALEMC composite because energy/carbon dominate.
* **Experiment 2 (config axis):** three RAG configurations (C1 lightweight /
  C2 balanced / C3 accuracy-first) on the phone, yielding an
  **accuracy–energy Pareto** with at least one ALEMC-dominated configuration.

## Layout

```
experiments/
  config.py                 # single source of truth: paths, models, grid γ, weight profiles
  requirements.txt          # python deps (graceful fallbacks if missing)
  data/
    prepare_queries.py      # freeze 100 NQ + 100 HotpotQA queries + gold answers
    prepare_corpus.py       # build the 20k-passage Wikipedia subset
    build_index.py          # embed corpus + build FAISS IndexFlatIP (run once, offline hub)
  pipeline/
    embedder.py             # query embedder (fastembed/ONNX -> sentence-transformers -> mock)
    backends.py             # generation backend (llama.cpp CLI -> llama-cpp-python -> mock)
    rag_pipeline.py         # per-query: embed -> search -> prompt -> generate; writes raw CSV
    run_experiment.py       # CLI: drives Exp1/Exp2 for one device+config
  samplers/
    sampler_windows.py      # X Elite laptop: WMI BatteryStatus.DischargeRate (mW)
    sampler_android.sh      # phone (Termux): /sys fuel-gauge current_now x voltage_now
    sampler_nvml.py         # RTX desktop: NVML / nvidia-smi power.draw
    energy_integrate.py     # trapezoidal integration, align samples to per-query windows
  scoring/
    score_accuracy.py       # EM, token-F1, RAGAS faithfulness -> accuracy harmonic mean
    composite.py            # ALEMC composite (eq:composite) with epsilon-smoothing
    sensitivity.py          # weight-profile + reference-set sensitivity
  analysis/
    aggregate.py            # raw CSV -> per-(device,config) means±std across 5 dims
    make_tables.py          # LaTeX tables: cross-tier, config, composite ranking
    make_figures.py         # data + PNG: accuracy-energy Pareto, ALEMC radar
  results/
    raw/                    # per-query CSV logs (one per device x config run)
    derived/                # scored + aggregated + composite outputs, LaTeX, figures
```

## End-to-end usage

### 0. Install (offline hub / any dev box)
```bash
pip install -r requirements.txt
```

### 1. Offline hub (RTX 5070 Ti) — build shared artifacts once
```bash
python data/prepare_queries.py        # -> data/queries.jsonl (frozen, 200 queries)
python data/prepare_corpus.py         # -> data/corpus.jsonl  (20k passages)
python data/build_index.py            # -> data/index.faiss + data/passages.npy
```
Ship `data/index.faiss`, `data/passages.*`, `data/queries.jsonl`, and the GGUF
model to each device.

### 2. On each device — run the pipeline while sampling energy
Start the device's energy sampler in one shell, run the pipeline in another;
they align by wall-clock timestamp.

```bash
# X Elite laptop (this repo's host machine), config = exp1 cross-tier
python samplers/sampler_windows.py --out results/raw/energy_laptop.csv &
python pipeline/run_experiment.py --device laptop --config exp1_baseline \
       --out results/raw/laptop_exp1.csv
```
See per-sampler headers for the phone (`sampler_android.sh`) and desktop
(`sampler_nvml.py`) invocations.

### 3. Offline hub — score, composite, sensitivity
```bash
python scoring/score_accuracy.py   results/raw/*.csv     # adds em,f1,faithfulness
python samplers/energy_integrate.py                      # folds energy into per-query rows
python analysis/aggregate.py                             # -> derived/aggregated.csv
python scoring/composite.py                              # -> derived/composite_*.csv
python scoring/sensitivity.py                            # -> derived/sensitivity_*.csv
python analysis/make_tables.py                           # -> derived/*.tex
python analysis/make_figures.py                          # -> derived/*.png + tikz data
```

## Design notes

* **Accuracy is device-independent.** It is a property of the model+pipeline
  *outputs*, scored centrally on the hub. For Experiment 1 (same model
  everywhere) accuracy is ~constant across devices, so cross-tier
  differentiation is purely L/E/M/C.
* **Embeddings are precomputed offline.** On-device work reduces to one small
  query-embed forward pass + a FAISS-CPU lookup + generation. This sidesteps a
  full PyTorch/sentence-transformers build on ARM/Termux — the biggest setup
  risk.
* **Graceful fallbacks.** Every backend degrades to a deterministic `mock`
  implementation when its native library is unavailable, so the harness and all
  downstream tables/figures can be validated without models present. Mock runs
  are flagged `is_mock=1` in the CSV and must never be reported as real data.

## Models

Model weights are **not** included in this repository (Gemma weights are
license-gated and the quantized GGUF is ~1.6 GB). See **[MODEL.md](MODEL.md)** for
how to obtain `gemma-2-2b-it` and quantize it to `Q4_K_M` with the pinned
`llama.cpp` commit, and how the `bge-small-en-v1.5` embedder is auto-fetched.

## License

- **Source code** (all `.py`, `.sh`): MIT — see [LICENSE](LICENSE).
- **Data** (`data/`, `results/`): CC BY-SA 4.0, inherited from SQuAD / HotpotQA /
  Wikipedia — see [DATA_LICENSE.md](DATA_LICENSE.md) for attribution.
- **Model weights**: not distributed; obtained under their own licenses (see
  [MODEL.md](MODEL.md)).

## Citation

If you use this code or data, please cite the accompanying paper and this
repository (see [CITATION.cff](CITATION.cff)).
