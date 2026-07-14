"""Build an *aligned* query set + corpus so every query is answerable.

Rationale (documented as a proof-of-concept limitation in the paper): the
frozen 20k-passage corpus is derived from HotpotQA + SQuAD contexts fetched via
the HF datasets-server REST API (the `datasets`/pyarrow stack has no ARM64
Windows wheel). Natural Questions gold answers live in full-Wikipedia articles
that are not in this scoped corpus, so open-domain NQ retrieval misses. To keep
retrieval coverage honest we use two QA sets whose gold passages ARE in the
corpus:

  * Factoid  : SQuAD v1.1 validation (question + short answer; its `context`
               paragraph is included in the corpus).
  * Multi-hop: HotpotQA distractor validation (its supporting `context`
               paragraphs are included in the corpus).

This mirrors the plan's "factoid + multi-hop" design (EXPERIMENT_PLAN §3) with
SQuAD substituted for NQ. Substitution is stated in the paper limitations.

Outputs (overwrites data/queries.jsonl and data/corpus.jsonl):
  queries.jsonl : {query_id,dataset,question,answers,supporting_titles}
  corpus.jsonl  : {id,title,text}  (gold passages first, then distractors)

Deterministic: takes the first N rows of each split; dedups corpus by text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (QUERIES_PATH, CORPUS_PATH, CORPUS_SIZE,
                    N_QUERIES_NQ, N_QUERIES_HOTPOT)
from data.hf_rest import fetch_rows


def _chunk(text, max_chars=600):
    text = " ".join(text.split())
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [text]


def build():
    queries = []
    corpus = []
    seen_text = set()

    def add_passage(title, text):
        for ch in _chunk(text):
            if len(ch) < 60:
                continue
            key = ch[:80]
            if key in seen_text:
                continue
            seen_text.add(key)
            corpus.append(dict(title=title, text=ch))

    # 1. SQuAD factoid queries + their gold contexts (guaranteed coverage).
    n = 0
    for row in fetch_rows("rajpurkar/squad", "plain_text", "validation",
                          N_QUERIES_NQ * 4):
        ans = row.get("answers", {})
        texts = ans.get("text", []) if isinstance(ans, dict) else []
        texts = list(dict.fromkeys([a for a in texts if a]))
        if not texts:
            continue
        title = row.get("title", "")
        add_passage(title, row.get("context", ""))
        queries.append(dict(query_id=f"squad-{n}", dataset="squad",
                            question=row["question"].strip(),
                            answers=texts, supporting_titles=[title]))
        n += 1
        if n >= N_QUERIES_NQ:
            break

    # 2. HotpotQA multi-hop queries + their supporting contexts.
    n = 0
    for row in fetch_rows("hotpotqa/hotpot_qa", "distractor", "validation",
                          N_QUERIES_HOTPOT * 4):
        ans = (row.get("answer") or "").strip()
        if not ans:
            continue
        ctx = row.get("context", {})
        titles = ctx.get("title", []) if isinstance(ctx, dict) else []
        sents = ctx.get("sentences", []) if isinstance(ctx, dict) else []
        for title, sl in zip(titles, sents):
            add_passage(title, " ".join(sl))
        sf = row.get("supporting_facts", {})
        sup = list(dict.fromkeys(sf.get("title", []))) if isinstance(sf, dict) else []
        queries.append(dict(query_id=f"hotpot-{n}", dataset="hotpotqa",
                            question=row["question"].strip(),
                            answers=[ans], supporting_titles=sup))
        n += 1
        if n >= N_QUERIES_HOTPOT:
            break

    # 3. Top up corpus with additional HotpotQA distractor contexts (each row
    #    contributes ~10 paragraphs, far more corpus-efficient than SQuAD and
    #    less likely to hit REST rate limits). These extra rows are used ONLY
    #    as distractor passages, not as queries.
    target = min(CORPUS_SIZE, 12_000)  # honest cap given REST availability
    if len(corpus) < target:
        for row in fetch_rows("hotpotqa/hotpot_qa", "distractor", "validation",
                              4000, page=100):
            ctx = row.get("context", {})
            titles = ctx.get("title", []) if isinstance(ctx, dict) else []
            sents = ctx.get("sentences", []) if isinstance(ctx, dict) else []
            for title, sl in zip(titles, sents):
                add_passage(title, " ".join(sl))
            if len(corpus) >= target:
                break
    corpus = corpus[:target]
    for i, p in enumerate(corpus):
        p["id"] = f"wiki-{i}"

    with open(QUERIES_PATH, "w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(CORPUS_PATH, "w", encoding="utf-8") as fh:
        for p in corpus:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    n_sq = sum(1 for q in queries if q["dataset"] == "squad")
    n_hp = sum(1 for q in queries if q["dataset"] == "hotpotqa")
    print(f"[dataset] queries={len(queries)} (squad={n_sq}, hotpot={n_hp}) "
          f"-> {QUERIES_PATH}")
    print(f"[dataset] corpus={len(corpus)} passages -> {CORPUS_PATH}")

    # coverage check
    corpus_titles = {p["title"] for p in corpus}
    covered = sum(1 for q in queries
                  if any(t in corpus_titles for t in q["supporting_titles"]))
    print(f"[dataset] queries with >=1 gold title in corpus: "
          f"{covered}/{len(queries)}")


if __name__ == "__main__":
    build()
