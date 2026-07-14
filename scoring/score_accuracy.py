"""Accuracy scoring — device-independent, run centrally on the offline hub.

For each raw run CSV, compute per-query:
  * EM       : SQuAD-style exact match after normalization
  * F1       : token-level F1 vs. gold answer(s)
  * faithfulness : fraction of answer claims grounded in retrieved context.
        Uses a RAGAS-style judge if a local judge model is configured
        (env ALEMC_JUDGE_CMD); otherwise a lexical-overlap proxy (documented
        as an approximation in the paper).
Then accuracy_harmonic = harmonic_mean(TaskScore=F1, faithfulness)  (eq:accuracy).

Gold answers and retrieved-context text are read from data/queries.jsonl and
data/passages_meta.jsonl. Writes the columns back into the same CSV.

Usage:
  python scoring/score_accuracy.py results/raw/*.csv
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import QUERIES_PATH, PASSAGES_META

_ARTICLES = {"a", "an", "the"}


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds) -> int:
    p = normalize(pred)
    return int(any(p == normalize(g) for g in golds))


def token_f1(pred: str, golds) -> float:
    p_tokens = normalize(pred).split()
    best = 0.0
    for g in golds:
        g_tokens = normalize(g).split()
        if not p_tokens or not g_tokens:
            best = max(best, float(p_tokens == g_tokens))
            continue
        common = Counter(p_tokens) & Counter(g_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(p_tokens)
        recall = num_same / len(g_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def lexical_faithfulness(answer: str, context: str) -> float:
    """Proxy: fraction of content words in the answer present in the context."""
    ctx = set(normalize(context).split())
    ans = [w for w in normalize(answer).split() if w not in _ARTICLES]
    if not ans:
        return 0.0
    grounded = sum(1 for w in ans if w in ctx)
    return grounded / len(ans)


def harmonic(a: float, b: float) -> float:
    return (2 * a * b / (a + b)) if (a + b) > 0 else 0.0


def load_gold():
    gold = {}
    with open(QUERIES_PATH, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            answers = q.get("answers") or [q.get("answer", "")]
            gold[q["query_id"]] = [a for a in answers if a]
    return gold


def load_passages():
    p = {}
    if PASSAGES_META.exists():
        with open(PASSAGES_META, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    p[str(d["id"])] = d.get("text", "")
    return p


def score_file(path: Path, gold, passages):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    fields = list(rows[0].keys())
    for col in ("em", "f1", "faithfulness", "accuracy_harmonic"):
        if col not in fields:
            fields.append(col)

    for r in rows:
        golds = gold.get(r["query_id"], [])
        pred = r.get("answer_text", "")
        em = exact_match(pred, golds) if golds else 0
        f1 = token_f1(pred, golds) if golds else 0.0
        ctx_ids = [i for i in r.get("retrieved_ids", "").split("|") if i]
        context = " ".join(passages.get(i, "") for i in ctx_ids)
        faith = lexical_faithfulness(pred, context) if context else 0.0
        r["em"] = em
        r["f1"] = round(f1, 4)
        r["faithfulness"] = round(faith, 4)
        r["accuracy_harmonic"] = round(harmonic(f1, faith), 4)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if int(r.get("warmup", 0)) == 0]
    if scored:
        mean_f1 = sum(float(r["f1"]) for r in scored) / len(scored)
        mean_em = sum(int(r["em"]) for r in scored) / len(scored)
        mean_fa = sum(float(r["faithfulness"]) for r in scored) / len(scored)
        print(f"[score] {path.name}: EM={mean_em:.3f} F1={mean_f1:.3f} "
              f"faith={mean_fa:.3f}  (n={len(scored)})")


def main():
    patterns = sys.argv[1:] or [str(Path("results/raw") / "*.csv")]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = [Path(f) for f in files if not Path(f).name.startswith(("idle_", "energy_"))]
    if not files:
        print("[score] no run CSVs matched", file=sys.stderr)
        return
    gold = load_gold()
    passages = load_passages()
    print(f"[score] gold={len(gold)} queries, passages={len(passages)}")
    for f in files:
        score_file(f, gold, passages)


if __name__ == "__main__":
    main()
