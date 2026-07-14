"""Build the 20,000-passage Wikipedia retrieval corpus.

Writes data/corpus.jsonl: {"id","title","text"} per line.

Prefers a Wikipedia passage dump via HuggingFace `datasets`
(`wiki_dpr` / `wikipedia`). To guarantee the frozen queries are answerable,
gold-supporting passages are always included when their source dataset exposes
context (HotpotQA distractor contexts), then topped up to CORPUS_SIZE with
general Wikipedia passages.

Offline fallback: a small synthetic corpus covering the fallback queries, so
retrieval + generation run end-to-end without network. Clearly tagged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CORPUS_PATH, CORPUS_SIZE, QUERIES_PATH


def _chunk(text, max_chars=600):
    text = " ".join(text.split())
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)] or [text]


def gold_passages_from_hotpot():
    """Pull the supporting contexts for the frozen HotpotQA queries."""
    out = []
    try:
        from datasets import load_dataset
        hp = load_dataset("hotpot_qa", "distractor", split="validation")
        seen = set()
        for ex in hp:
            ctx = ex.get("context", {})
            titles = ctx.get("title", [])
            sents = ctx.get("sentences", [])
            for title, sent_list in zip(titles, sents):
                key = title
                if key in seen:
                    continue
                seen.add(key)
                text = " ".join(sent_list)
                for j, ch in enumerate(_chunk(text)):
                    out.append(dict(title=title, text=ch))
            if len(out) >= CORPUS_SIZE:
                break
    except Exception as e:
        print(f"[corpus] hotpot context load failed: {e}", file=sys.stderr)
    return out


def general_wikipedia(need):
    out = []
    for cfg in [("wikipedia", "20220301.simple"), ("wikipedia", "20220301.en")]:
        try:
            from datasets import load_dataset
            ds = load_dataset(cfg[0], cfg[1], split="train", streaming=True)
            for ex in ds:
                for ch in _chunk(ex.get("text", "")):
                    if len(ch) < 100:
                        continue
                    out.append(dict(title=ex.get("title", ""), text=ch))
                    if len(out) >= need:
                        return out
        except Exception as e:
            print(f"[corpus] {cfg} failed: {e}", file=sys.stderr)
            continue
    return out


def from_hf_rest(need):
    """Build the corpus via the REST API (no pyarrow).

    HotpotQA `context` supplies gold-supporting Wikipedia paragraphs (ensuring
    the frozen queries are answerable); SQuAD `context` tops up with general
    Wikipedia passages. Deduplicated by (title, text-prefix).
    """
    from data.hf_rest import fetch_rows
    out, seen = [], set()

    # 1. HotpotQA supporting contexts (gold coverage for the multi-hop queries)
    try:
        for row in fetch_rows("hotpotqa/hotpot_qa", "distractor",
                              "validation", 2000):
            ctx = row.get("context", {})
            titles = ctx.get("title", []) if isinstance(ctx, dict) else []
            sents = ctx.get("sentences", []) if isinstance(ctx, dict) else []
            for title, sent_list in zip(titles, sents):
                text = " ".join(sent_list)
                key = (title, text[:60])
                if key in seen or len(text) < 80:
                    continue
                seen.add(key)
                for ch in _chunk(text):
                    out.append(dict(title=title, text=ch))
            if len(out) >= need:
                return out[:need]
    except Exception as e:
        print(f"[corpus] hotpot REST failed: {e}", file=sys.stderr)

    # 2. SQuAD contexts as general Wikipedia filler
    try:
        for row in fetch_rows("rajpurkar/squad", "plain_text", "validation", need):
            title = row.get("title", "")
            text = row.get("context", "")
            key = (title, text[:60])
            if key in seen or len(text) < 80:
                continue
            seen.add(key)
            for ch in _chunk(text):
                out.append(dict(title=title, text=ch))
            if len(out) >= need:
                break
    except Exception as e:
        print(f"[corpus] squad REST failed: {e}", file=sys.stderr)
    return out[:need]


FALLBACK_CORPUS = [
    ("William Shakespeare", "William Shakespeare was an English playwright and poet, widely regarded as the greatest writer in the English language. He wrote the play Romeo and Juliet, a tragedy about two young star-crossed lovers."),
    ("Paris", "Paris is the capital and most populous city of France. It is situated on the river Seine in northern France."),
    ("Gold", "Gold is a chemical element with the symbol Au and atomic number 79. It is a dense, soft, precious metal used in jewellery and electronics."),
    ("Mona Lisa", "The Mona Lisa is a half-length portrait painting by the Italian polymath Leonardo da Vinci. It is considered an archetypal masterpiece of the Italian Renaissance."),
    ("Mars", "Mars is the fourth planet from the Sun and is known as the Red Planet because of its reddish appearance caused by iron oxide on its surface."),
    ("World War II", "World War II was a global conflict that lasted from 1939 to 1945. It involved the Allied powers and the Axis powers and ended in 1945 with the surrender of the Axis."),
    ("Pacific Ocean", "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south."),
    ("Albert Einstein", "Albert Einstein was a German-born theoretical physicist who developed the theory of general relativity and is famous for the mass-energy equivalence formula E=mc^2."),
    ("Mount Everest", "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas. Its peak is the highest point on the planet."),
    ("Photosynthesis", "Photosynthesis is the process by which green plants absorb carbon dioxide from the atmosphere and use sunlight to synthesize nutrients, releasing oxygen."),
]


def from_fallback():
    out = []
    # duplicate with slight variety to create a non-trivial retrieval set
    for i, (title, text) in enumerate(FALLBACK_CORPUS):
        out.append(dict(id=f"fb-{i}", title=title, text=text, _fallback=True))
    # distractors
    distract = [
        "The mitochondrion is the powerhouse of the cell in eukaryotic organisms.",
        "The Great Wall of China is a series of fortifications built across northern China.",
        "Python is a high-level, general-purpose programming language.",
        "The Amazon rainforest is a moist broadleaf tropical rainforest in South America.",
        "Basketball is a team sport in which two teams score by shooting a ball through a hoop.",
    ]
    for j, d in enumerate(distract):
        out.append(dict(id=f"fb-d{j}", title=f"Distractor {j}", text=d, _fallback=True))
    return out


def main():
    passages = gold_passages_from_hotpot()
    if len(passages) < CORPUS_SIZE:
        passages += general_wikipedia(CORPUS_SIZE - len(passages))

    # REST fallback when `datasets` is unavailable (ARM64 host).
    if len(passages) < CORPUS_SIZE:
        try:
            rest = from_hf_rest(CORPUS_SIZE - len(passages))
            if rest:
                print(f"[corpus] fetched {len(rest)} passages via HF REST API")
                passages += rest
        except Exception as e:
            print(f"[corpus] REST corpus build failed: {e}", file=sys.stderr)

    if len(passages) < 50:
        print("[corpus] WARNING: could not build Wikipedia corpus; using "
              "bundled FALLBACK corpus — NOT the 20k benchmark corpus.",
              file=sys.stderr)
        passages = from_fallback()
    else:
        passages = passages[:CORPUS_SIZE]
        for i, p in enumerate(passages):
            p["id"] = f"wiki-{i}"

    with open(CORPUS_PATH, "w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[corpus] wrote {len(passages)} passages -> {CORPUS_PATH}")


if __name__ == "__main__":
    main()
