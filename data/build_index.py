"""Embed the corpus and build a FAISS IndexFlatIP (run once, offline hub).

Reads data/corpus.jsonl, embeds every passage with the shared embedder,
writes:
  data/index.faiss        (FAISS IndexFlatIP over L2-normalized vectors)
  data/passages.npy       (float32 vectors, fallback if FAISS unavailable)
  data/passages_meta.jsonl (id/title/text aligned to vector rows)

Ship these three files to each device; on-device work is then just a query
embed + FAISS lookup + generation (EXPERIMENT_PLAN §3).

Usage:
  python data/build_index.py [--batch 256]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CORPUS_PATH, INDEX_PATH, PASSAGES_PATH, PASSAGES_META, EMBED_DIM
from pipeline.embedder import Embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    if not CORPUS_PATH.exists():
        sys.exit(f"[index] missing {CORPUS_PATH}; run data/prepare_corpus.py first")

    passages = []
    with open(CORPUS_PATH, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                passages.append(json.loads(line))
    print(f"[index] {len(passages)} passages")

    emb = Embedder()
    print(f"[index] embedder backend = {emb.backend}")
    if emb.is_mock:
        print("[index] WARNING: MOCK embedder — index is not semantically "
              "meaningful; for pipeline validation only.")

    texts = [p.get("text", "") for p in passages]
    vecs = np.zeros((len(texts), emb.dim), dtype=np.float32)
    for i in range(0, len(texts), args.batch):
        chunk = texts[i:i + args.batch]
        vecs[i:i + len(chunk)] = emb.encode(chunk)
        if (i // args.batch) % 10 == 0:
            print(f"  embedded {min(i + len(chunk), len(texts))}/{len(texts)}")

    # meta aligned to rows
    with open(PASSAGES_META, "w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(dict(id=p.get("id"), title=p.get("title", ""),
                                     text=p.get("text", "")),
                                ensure_ascii=False) + "\n")

    np.save(PASSAGES_PATH, vecs)
    print(f"[index] saved vectors -> {PASSAGES_PATH}  shape={vecs.shape}")

    try:
        import faiss
        index = faiss.IndexFlatIP(emb.dim)
        index.add(vecs)
        faiss.write_index(index, str(INDEX_PATH))
        print(f"[index] wrote FAISS IndexFlatIP -> {INDEX_PATH}  "
              f"ntotal={index.ntotal}")
    except Exception as e:
        print(f"[index] FAISS unavailable ({e}); retriever will use "
              f"numpy fallback on {PASSAGES_PATH}")


if __name__ == "__main__":
    main()
