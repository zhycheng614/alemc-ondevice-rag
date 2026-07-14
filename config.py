"""Single source of truth for the ALEMC empirical study.

All paths, model identifiers, grid carbon intensities, and ALEMC weight
profiles live here so every script agrees. Values mirror EXPERIMENT_PLAN.md
§3 (shared setup) and 07_convergence.tex (equations and weight profiles).
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw"
DERIVED_DIR = RESULTS_DIR / "derived"
MODELS_DIR = ROOT / "models"

for _d in (DATA_DIR, RAW_DIR, DERIVED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

QUERIES_PATH = DATA_DIR / "queries.jsonl"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
INDEX_PATH = DATA_DIR / "index.faiss"
PASSAGES_PATH = DATA_DIR / "passages.npy"       # (N, dim) float32 passage vectors
PASSAGES_META = DATA_DIR / "passages_meta.jsonl"  # id/title/text per row, aligned to vectors

# ----------------------------------------------------------------------------
# Shared experimental setup (EXPERIMENT_PLAN.md §3)
# ----------------------------------------------------------------------------
# Generator: Gemma-2-2B-it, 4-bit Q4_K_M GGUF.
GEN_MODEL_NAME = "gemma-2-2b-it"
GEN_MODEL_GGUF = os.environ.get(
    "ALEMC_GGUF",
    str(MODELS_DIR / "gemma-2-2b-it-Q4_K_M.gguf"),
)
# A small, reliable fallback model for smoke tests / constrained laptops.
GEN_MODEL_GGUF_FALLBACK = os.environ.get("ALEMC_GGUF_FALLBACK", "")

# Embedder: bge-small-en-v1.5 (384-dim). fastembed uses the ONNX version and
# needs no torch, which is why it is preferred for on-device query encoding.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# Corpus / queries
CORPUS_SIZE = 20_000
N_QUERIES_NQ = 100
N_QUERIES_HOTPOT = 100
RETRIEVER_K_DEFAULT = 5

# Decode params (fixed for determinism, EXPERIMENT_PLAN.md §3)
MAX_TOKENS = 256
TEMPERATURE = 0.0            # greedy
WARMUP_QUERIES = 5

# Prompt template (fixed across all runs)
PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "below. Give a short, direct answer. If the context does not contain the "
    "answer, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# ----------------------------------------------------------------------------
# Experiment 2 configurations (EXPERIMENT_PLAN.md §5)
# ----------------------------------------------------------------------------
# Each config selects retrieval depth, whether to use a reranker, and whether
# to apply context compression. Archetypes map to surveyed systems.
CONFIGS = {
    # Experiment 1 uses the same baseline pipeline on every device.
    "exp1_baseline": dict(k=5, reranker=False, compress=False,
                          archetype="cross-tier baseline"),
    # Experiment 2 config axis (phone only).
    "C1": dict(k=3, reranker=False, compress=False,
               archetype="MiniRAG-style lightweight"),
    "C2": dict(k=5, reranker=False, compress=True,
               archetype="EdgeRAG/MobileRAG-style balanced"),
    "C3": dict(k=10, reranker=True, compress=False,
               archetype="PocketRAG-style accuracy-first"),
}

DEVICES = {
    "phone":   dict(label="Snapdragon 8 Elite", tier="mobile"),
    "laptop":  dict(label="Snapdragon X Elite", tier="edge"),
    "desktop": dict(label="RTX 5070 Ti",        tier="desktop"),
}

# ----------------------------------------------------------------------------
# Carbon (07_convergence.tex eq:carbon): C = E[J] * gamma[gCO2/kWh] * 1e-3 / 3600
# Reported at three reference grid intensities (paper §7.2.5 / EXPERIMENT_PLAN §6.5)
# ----------------------------------------------------------------------------
GRID_INTENSITIES = {"low": 50.0, "medium": 300.0, "high": 600.0}  # gCO2 / kWh


def joules_to_gco2e(energy_j: float, gamma_g_per_kwh: float) -> float:
    """eq:carbon — J and gCO2/kWh -> gCO2e per query."""
    return energy_j * gamma_g_per_kwh * 1e-3 / 3600.0


# ----------------------------------------------------------------------------
# ALEMC composite weight profiles (07_convergence.tex:147)
# order: (accuracy, latency, energy, memory, carbon); sum to 1
# ----------------------------------------------------------------------------
WEIGHT_PROFILES = {
    "accuracy-first":   dict(a=0.40, l=0.20, e=0.15, m=0.15, c=0.10),
    "efficiency-first": dict(a=0.20, l=0.15, e=0.30, m=0.15, c=0.20),
    "balanced":         dict(a=0.25, l=0.20, e=0.20, m=0.15, c=0.20),
}

# epsilon-smoothing for the composite (EXPERIMENT_PLAN §9 / R2-minor-2):
# normalize each dimension to [DELTA, 1] instead of [0, 1] so a worst-in-set
# value cannot produce a zero base under a negative exponent.
COMPOSITE_DELTA = 0.05
