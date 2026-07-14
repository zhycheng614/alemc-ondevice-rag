# Data License — CC BY-SA 4.0

The **data artifacts** in this repository are licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

This covers, in particular:

- `data/queries.jsonl`, `data/queries_phone.jsonl` — the frozen evaluation query sets
- `data/corpus.jsonl`, `data/passages_meta.jsonl` — the 12,000-passage retrieval corpus
- `data/index.faiss`, `data/passages.npy` — the FAISS index and passage embeddings derived from the corpus
- `results/` — raw per-query logs and derived tables/figures computed from the above

(The **source code** in this repository is licensed separately under the MIT License; see `LICENSE`.)

## Attribution of upstream sources

The corpus and query sets are derived from the following datasets, each distributed
under CC BY-SA 4.0. Their ShareAlike terms are why the derived data here is also
CC BY-SA 4.0. Please cite them if you use this data:

- **SQuAD v1.1** — Rajpurkar, Zhang, Lopyrev, Liang. *SQuAD: 100,000+ Questions for
  Machine Comprehension of Text.* EMNLP 2016. (Passages/questions derived from
  English Wikipedia.)
- **HotpotQA** — Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning.
  *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.*
  EMNLP 2018. (Supporting contexts derived from English Wikipedia.)
- **Wikipedia** — text content © its contributors, available under CC BY-SA
  (via the SQuAD and HotpotQA context paragraphs).

If you redistribute or build upon these data files, you must retain attribution
and license derivatives under the same CC BY-SA 4.0 terms.
