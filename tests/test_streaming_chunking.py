import numpy as np

from quint.chunking.streaming import (
    merge_short_sentences,
    stream_boundaries,
    stream_chunks,
)


def _synthetic_harness(n_topics=7, per=6, alpha=0.6, seed=0, dim=64):
    """Concatenated 'passages' with known seams.

    Each topic has a base vector; each sentence is alpha*base + (1-alpha)*noise,
    so within-topic similarity is high and cross-topic is ~0 — a ground truth to
    score boundary detection against.
    """
    rng = np.random.default_rng(seed)
    bases = rng.normal(size=(n_topics, dim))
    vecs, seams = [], set()
    for t in range(n_topics):
        if vecs:
            seams.add(len(vecs))
        for _ in range(per):
            v = alpha * bases[t] + (1 - alpha) * rng.normal(size=dim)
            vecs.append(v / np.linalg.norm(v))
    return np.array(vecs), seams


def _f1(pred, true, tol=1):
    pred = set(pred)
    if not pred:
        return 0.0
    prec = sum(any(abs(p - t) <= tol for t in true) for p in pred) / len(pred)
    rec = sum(any(abs(t - p) <= tol for p in pred) for t in true) / len(true)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def test_boundary_f1_clears_0_9():
    """The improved chunker recovers known topic seams with F1 >= 0.9."""
    V, seams = _synthetic_harness()
    sents = list(range(len(V)))
    best = max(
        _f1(stream_boundaries(sents, lambda i: V[i], threshold=th, min_size=2), seams)
        for th in np.arange(-0.1, 0.5, 0.02)
    )
    assert best >= 0.9, f"best F1 only {best:.2f}"


def test_no_singleton_chunks():
    """min_size=2 structurally forbids one-sentence chunks (the V0 flaw)."""
    V, _ = _synthetic_harness()
    bounds = stream_boundaries(list(range(len(V))), lambda i: V[i], 0.2, min_size=2)
    edges = [0] + bounds + [len(V)]
    sizes = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    assert min(sizes) >= 2, sizes


def test_splits_two_clear_topics():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    V = [a, a, a, b, b, b]
    assert stream_boundaries(list(range(6)), lambda i: V[i], 0.5, min_size=2) == [3]


def test_merge_short_sentences():
    out = merge_short_sentences(
        ["This is a normal sentence here.", "Yes.", "Another normal one follows."],
        min_words=4,
    )
    assert out == ["This is a normal sentence here. Yes.", "Another normal one follows."]


def test_handles_short_input():
    assert stream_chunks(["Only one sentence here."], lambda s: np.ones(8)) == [
        "Only one sentence here."
    ]
