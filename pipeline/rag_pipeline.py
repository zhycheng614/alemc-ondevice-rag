"""The device-agnostic RAG pipeline.

Per query: query-embed -> FAISS top-k search -> (optional rerank) ->
(optional context compression) -> prompt assembly -> generation.

Timing is instrumented so eq:latency components (t_retrieval, t_prefill,
t_decode, n_tokens) are all logged. Memory peak is captured via memtools.
Energy is measured *out of process* by the per-device sampler and folded in
later by samplers/energy_integrate.py, aligned on the query time window
(t_query_start / t_query_end wall-clock epoch seconds, logged here).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (INDEX_PATH, PASSAGES_META, PROMPT_TEMPLATE, CONFIGS,
                    RETRIEVER_K_DEFAULT)
from pipeline.embedder import Embedder
from pipeline.backends import Generator
from pipeline.memtools import peak_rss_mb, external_process_rss_mb


@dataclass
class QueryRecord:
    query_id: str
    dataset: str
    # latency
    ttft_ms: float
    e2e_ms: float
    t_retrieval_ms: float
    t_prefill_ms: float
    t_decode_ms: float
    n_tokens: int
    # memory
    peak_rss_mb: float
    # energy window (epoch seconds; sampler aligns to this)
    t_query_start: float
    t_query_end: float
    # output
    answer_text: str
    retrieved_ids: str
    is_mock: int


class Retriever:
    """FAISS IndexFlatIP brute-force over precomputed passage vectors."""

    def __init__(self):
        self.passages: List[dict] = []
        self._index = None
        self._vectors = None
        self._load()

    def _load(self):
        if PASSAGES_META.exists():
            with open(PASSAGES_META, encoding="utf-8") as fh:
                self.passages = [json.loads(l) for l in fh if l.strip()]
        # Prefer a prebuilt FAISS index; fall back to numpy vectors; else empty.
        try:
            import faiss
            if INDEX_PATH.exists():
                self._index = faiss.read_index(str(INDEX_PATH))
        except Exception:
            self._index = None
        if self._index is None:
            from config import PASSAGES_PATH
            if PASSAGES_PATH.exists():
                self._vectors = np.load(PASSAGES_PATH).astype(np.float32)

    @property
    def ready(self) -> bool:
        return bool(self.passages) and (self._index is not None
                                        or self._vectors is not None)

    def search(self, qvec: np.ndarray, k: int):
        qvec = qvec.reshape(1, -1).astype(np.float32)
        if self._index is not None:
            scores, idx = self._index.search(qvec, k)
            idx = idx[0]
        elif self._vectors is not None:
            sims = (self._vectors @ qvec[0])
            idx = np.argsort(-sims)[:k]
        else:
            idx = np.arange(min(k, len(self.passages)))
        return [self.passages[i] for i in idx if 0 <= i < len(self.passages)]


def _compress_context(passages: List[dict], max_chars: int = 900) -> str:
    """C2 context compression: keep leading sentences up to a char budget."""
    out, used = [], 0
    for p in passages:
        text = p.get("text", "")
        if used + len(text) > max_chars:
            text = text[: max(0, max_chars - used)]
        out.append(text)
        used += len(text)
        if used >= max_chars:
            break
    return "\n".join(out)


def _rerank(query: str, passages: List[dict], embedder: Embedder,
            top: int) -> List[dict]:
    """C3 lightweight cross-encoder-style rerank.

    If a real cross-encoder (bge-reranker) is available it is used; otherwise
    we fall back to a lexical-overlap rerank so the config still differentiates
    (documented as a proxy in the paper's limitations).
    """
    try:
        from sentence_transformers import CrossEncoder
        ce = _rerank._ce  # cache
        if ce is None:
            ce = CrossEncoder("BAAI/bge-reranker-base")
            _rerank._ce = ce
        scores = ce.predict([(query, p.get("text", "")) for p in passages])
        order = np.argsort(-np.asarray(scores))
        return [passages[i] for i in order[:top]]
    except Exception:
        q_terms = set(query.lower().split())
        def overlap(p):
            terms = set(p.get("text", "").lower().split())
            return len(q_terms & terms)
        return sorted(passages, key=overlap, reverse=True)[:top]
_rerank._ce = None


class RAGPipeline:
    def __init__(self, config_name: str = "exp1_baseline",
                 embedder: Optional[Embedder] = None,
                 generator: Optional[Generator] = None):
        self.config_name = config_name
        self.cfg = CONFIGS[config_name]
        self.embedder = embedder or Embedder()
        self.generator = generator or Generator()
        self.retriever = Retriever()

    @property
    def is_mock(self) -> bool:
        return self.embedder.is_mock or self.generator.is_mock

    def run_query(self, query_id: str, question: str, dataset: str) -> QueryRecord:
        cfg = self.cfg
        t_start = time.time()

        # 1. retrieval (embed + search [+ rerank])
        t_r0 = time.perf_counter()
        qvec = self.embedder.encode([question])[0]
        fetch_k = cfg["k"] * 3 if cfg["reranker"] else cfg["k"]
        passages = self.retriever.search(qvec, fetch_k)
        if cfg["reranker"]:
            passages = _rerank(question, passages, self.embedder, cfg["k"])
        t_retrieval_ms = (time.perf_counter() - t_r0) * 1e3

        # 2. prompt assembly (+ optional compression)
        if cfg["compress"]:
            context = _compress_context(passages)
        else:
            context = "\n".join(p.get("text", "") for p in passages)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # 3. generation
        gen = self.generator.generate(prompt)
        t_end = time.time()

        # Memory: pipeline RSS + resident generation-server RSS (model weights +
        # KV cache live in llama-server when the server backend is used). On a
        # single-process deployment external_rss is 0 and this is just RSS.
        # For the adb (phone) backend, memory is the on-device llama-cli peak,
        # captured separately per run (see run_phone_tier); here we record the
        # host pipeline RSS only (retrieval/embedding footprint).
        mem_mb = peak_rss_mb()
        if self.generator.backend == "llama.cpp-server":
            mem_mb += external_process_rss_mb("llama-server")

        return QueryRecord(
            query_id=query_id, dataset=dataset,
            ttft_ms=round(gen.ttft_ms + t_retrieval_ms, 3),  # TTFT includes retrieval
            e2e_ms=round(gen.e2e_ms + t_retrieval_ms, 3),
            t_retrieval_ms=round(t_retrieval_ms, 3),
            t_prefill_ms=round(gen.t_prefill_ms, 3),
            t_decode_ms=round(gen.t_decode_ms, 4),
            n_tokens=gen.n_tokens,
            peak_rss_mb=round(mem_mb, 1),
            t_query_start=t_start, t_query_end=t_end,
            answer_text=gen.text.replace("\n", " ").strip(),
            retrieved_ids="|".join(str(p.get("id", "")) for p in passages),
            is_mock=int(self.is_mock),
        )


if __name__ == "__main__":
    p = RAGPipeline("exp1_baseline")
    print(f"embedder={p.embedder.backend} generator={p.generator.backend} "
          f"retriever_ready={p.retriever.ready} mock={p.is_mock}")
    rec = p.run_query("demo-1", "What is the capital of France?", "demo")
    print(asdict(rec))
