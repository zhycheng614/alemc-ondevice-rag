"""Freeze the 200-query evaluation set: 100 Natural Questions + 100 HotpotQA.

Writes data/queries.jsonl, one JSON per line:
  {"query_id","dataset","question","answers":[...],"supporting_titles":[...]}

Prefers HuggingFace `datasets`. If unavailable or offline, falls back to a
small bundled sample so the harness is runnable end-to-end; a WARNING is
printed and the file is tagged so it is never mistaken for the frozen set.

Deterministic: takes the first N validation examples with non-empty short
answers (no RNG), so re-running reproduces the identical set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import QUERIES_PATH, N_QUERIES_NQ, N_QUERIES_HOTPOT


def from_hf():
    from datasets import load_dataset
    out = []

    # Natural Questions (open-domain short answers). Use the 'nq_open' subset
    # which provides clean question/answer pairs.
    try:
        nq = load_dataset("nq_open", split="validation")
        n = 0
        for ex in nq:
            answers = [a for a in ex.get("answer", []) if a]
            if not answers:
                continue
            out.append(dict(query_id=f"nq-{n}", dataset="nq",
                            question=ex["question"].strip(),
                            answers=answers, supporting_titles=[]))
            n += 1
            if n >= N_QUERIES_NQ:
                break
    except Exception as e:
        print(f"[queries] NQ load failed: {e}", file=sys.stderr)

    # HotpotQA (multi-hop).
    try:
        hp = load_dataset("hotpot_qa", "distractor", split="validation")
        n = 0
        for ex in hp:
            ans = ex.get("answer", "").strip()
            if not ans:
                continue
            titles = list(dict.fromkeys(ex.get("supporting_facts", {}).get("title", [])))
            out.append(dict(query_id=f"hotpot-{n}", dataset="hotpotqa",
                            question=ex["question"].strip(),
                            answers=[ans], supporting_titles=titles))
            n += 1
            if n >= N_QUERIES_HOTPOT:
                break
    except Exception as e:
        print(f"[queries] HotpotQA load failed: {e}", file=sys.stderr)

    return out


def from_hf_rest():
    """Fetch NQ + HotpotQA via the datasets-server REST API (no pyarrow)."""
    from data.hf_rest import fetch_rows
    out = []
    n = 0
    for row in fetch_rows("google-research-datasets/nq_open", "nq_open",
                          "validation", N_QUERIES_NQ * 2):
        answers = row.get("answer") or []
        answers = [a for a in (answers if isinstance(answers, list) else [answers]) if a]
        if not answers:
            continue
        out.append(dict(query_id=f"nq-{n}", dataset="nq",
                        question=row["question"].strip(),
                        answers=answers, supporting_titles=[]))
        n += 1
        if n >= N_QUERIES_NQ:
            break
    n = 0
    for row in fetch_rows("hotpotqa/hotpot_qa", "distractor",
                          "validation", N_QUERIES_HOTPOT * 2):
        ans = (row.get("answer") or "").strip()
        if not ans:
            continue
        sf = row.get("supporting_facts", {})
        titles = list(dict.fromkeys(sf.get("title", []))) if isinstance(sf, dict) else []
        out.append(dict(query_id=f"hotpot-{n}", dataset="hotpotqa",
                        question=row["question"].strip(),
                        answers=[ans], supporting_titles=titles))
        n += 1
        if n >= N_QUERIES_HOTPOT:
            break
    return out


# Minimal offline fallback (10 factoid + 10 multi-hop-style) so the pipeline
# is demonstrably runnable without network. Clearly tagged.
FALLBACK = [
    ("nq", "Who wrote the play Romeo and Juliet?", ["William Shakespeare"]),
    ("nq", "What is the capital of France?", ["Paris"]),
    ("nq", "What is the chemical symbol for gold?", ["Au"]),
    ("nq", "Who painted the Mona Lisa?", ["Leonardo da Vinci"]),
    ("nq", "What planet is known as the Red Planet?", ["Mars"]),
    ("nq", "In what year did World War II end?", ["1945"]),
    ("nq", "What is the largest ocean on Earth?", ["Pacific Ocean", "the Pacific"]),
    ("nq", "Who developed the theory of general relativity?", ["Albert Einstein"]),
    ("nq", "What is the tallest mountain on Earth?", ["Mount Everest"]),
    ("nq", "What gas do plants absorb from the atmosphere?", ["carbon dioxide", "CO2"]),
    ("hotpotqa", "What nationality was the author of Romeo and Juliet?", ["English"]),
    ("hotpotqa", "In which country is the capital city Paris located?", ["France"]),
    ("hotpotqa", "What element has the chemical symbol Au and is a precious metal?", ["gold"]),
    ("hotpotqa", "Which Italian polymath painted the Mona Lisa?", ["Leonardo da Vinci"]),
    ("hotpotqa", "What is the fourth planet from the Sun?", ["Mars"]),
    ("hotpotqa", "Which war ended in 1945 and involved the Allied powers?", ["World War II"]),
    ("hotpotqa", "What is the deepest and largest ocean on the planet?", ["Pacific Ocean"]),
    ("hotpotqa", "Who is the physicist behind E=mc^2 and general relativity?", ["Albert Einstein"]),
    ("hotpotqa", "What Himalayan peak is the highest above sea level?", ["Mount Everest"]),
    ("hotpotqa", "What gas, used in photosynthesis, is absorbed by green plants?", ["carbon dioxide"]),
]


def from_fallback():
    out = []
    for i, (ds, q, ans) in enumerate(FALLBACK):
        out.append(dict(query_id=f"{ds}-fb-{i}", dataset=ds, question=q,
                        answers=ans, supporting_titles=[], _fallback=True))
    return out


def main():
    queries = []
    try:
        queries = from_hf()
    except Exception as e:
        print(f"[queries] `datasets` unavailable ({e}); trying REST API",
              file=sys.stderr)
    if len(queries) < 20:
        try:
            queries = from_hf_rest()
            print(f"[queries] fetched {len(queries)} via HF REST API")
        except Exception as e:
            print(f"[queries] REST API failed ({e}); using bundled fallback",
                  file=sys.stderr)
            queries = []
    if len(queries) < 20:
        print("[queries] WARNING: insufficient HF queries; using bundled "
              "FALLBACK set — NOT the frozen 200-query benchmark set.",
              file=sys.stderr)
        queries = from_fallback()

    with open(QUERIES_PATH, "w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    n_nq = sum(1 for q in queries if q["dataset"] == "nq")
    n_hp = sum(1 for q in queries if q["dataset"] == "hotpotqa")
    print(f"[queries] wrote {len(queries)} queries (nq={n_nq}, hotpot={n_hp}) "
          f"-> {QUERIES_PATH}")


if __name__ == "__main__":
    main()
