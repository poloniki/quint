"""Graded semantic-chunking benchmark: easy (cross-domain) vs hard (same-domain).

Compares three thresholding strategies of the streaming chunker:
- fixed @ default (0.12): one baked-in constant — brittle off its tuned corpus.
- fixed @ best-tuned: the per-corpus best (needs the whole document — not streaming).
- adaptive (z): self-calibrating, scale-free, no prior threshold.

Run with a model2vec-capable env (see notebooks/ for the build venv).
"""
import sys
import json
from collections import defaultdict

import numpy as np
import pysbd
from model2vec import StaticModel

sys.path.insert(0, ".")
from quint.chunking.streaming import stream_boundaries, stream_boundaries_adaptive

seg = pysbd.Segmenter(language="en", clean=False)
corpus = json.load(open("benchmarks/chunking/corpus.json"))
byc = defaultdict(list)
for p in corpus:
    byc[p["cluster"]].append(p)
m = StaticModel.from_pretrained("minishlab/potion-base-8M")


def embed(s):
    v = np.asarray(m.encode(s), np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def make_doc(passages):
    sents, seams = [], set()
    for p in passages:
        s = seg.segment(p["text"])
        if sents:
            seams.add(len(sents))
        sents += s
    return np.array([embed(x) for x in sents]), seams, len(sents)


clusters = list(byc)
hard = [make_doc(byc[c]) for c in clusters]
rng = np.random.default_rng(0)
easy = [
    make_doc([byc[clusters[i]][int(rng.integers(0, 3))]
              for i in rng.choice(len(clusters), 4, replace=False)])
    for _ in range(8)
]


def f1(pred, true, tol=1):
    P = set(pred)
    if not P:
        return 0.0
    pr = sum(any(abs(p - t) <= tol for t in true) for p in P) / len(P)
    rc = sum(any(abs(t - p) <= tol for p in P) for t in true) / len(true)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def mean_f1(docs, boundary_fn):
    return float(np.mean([f1(boundary_fn(V), seams) for V, seams, N in docs]))


METHODS = {
    "fixed @ default 0.12": lambda docs: mean_f1(
        docs, lambda V: stream_boundaries(list(range(len(V))), lambda i: V[i], 0.12, 2)
    ),
    "fixed @ best-tuned": lambda docs: max(
        mean_f1(docs, lambda V, t=t: stream_boundaries(list(range(len(V))), lambda i: V[i], t, 2))
        for t in np.arange(0.0, 0.40, 0.02)
    ),
    "adaptive z=1.3": lambda docs: mean_f1(
        docs, lambda V: stream_boundaries_adaptive(list(range(len(V))), lambda i: V[i], 1.3)
    ),
}

print(f"{'method':22} {'easy F1':>8} {'hard F1':>8}")
for name, fn in METHODS.items():
    print(f"{name:22} {fn(easy):8.2f} {fn(hard):8.2f}")
