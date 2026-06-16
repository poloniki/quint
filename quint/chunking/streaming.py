"""Single-pass, streaming semantic chunking.

Splits text into semantically coherent chunks in one forward pass — no O(n^2)
similarity matrix and no lookahead — so it works on a live stream. For each
sentence it compares the embedding to a running centroid of the current chunk
and opens a new chunk when the topic drifts away.

The split logic is embedding-agnostic: pass any callable mapping a sentence to a
vector (it is L2-normalized internally). See quint.chunking.generate.get_chunks
for the original matrix-based method.

Two boundary rules are provided:

- ``stream_chunks`` uses a **fixed** similarity threshold. Simple and fast, but
  the right value depends on the embedding model *and the corpus*, so it does
  not transfer well between very different text types.
- ``stream_chunks_adaptive`` uses an **adaptive** threshold: it splits when a
  sentence is anomalously dissimilar relative to the running mean/spread of
  similarities seen *so far in the stream* (``sim < mean - z*std``). The knob
  ``z`` is scale-free (standard deviations), needs no prior calibration, and
  self-adjusts to each document — the right choice for true streaming.

Both share two quality guards: ``min_size`` forbids closing a chunk before it
has that many sentences (kills spurious single-sentence chunks), and
``merge_short`` folds very short fragments into their neighbour first.
"""

import numpy as np


def merge_short_sentences(sentences, min_words=4):
    """Fold sentences shorter than ``min_words`` words into the preceding one."""
    merged = []
    for s in sentences:
        if merged and len(s.split()) < min_words:
            merged[-1] = f"{merged[-1]} {s}"
        else:
            merged.append(s)
    return merged


def _normalized(embed, s):
    v = np.asarray(embed(s), dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def _build_chunks(sentences, boundaries):
    chunks, prev = [], 0
    for b in list(boundaries) + [len(sentences)]:
        chunk = " ".join(sentences[prev:b]).strip()
        if chunk:
            chunks.append(chunk)
        prev = b
    return chunks


def stream_boundaries(sentences, embed, threshold=0.12, min_size=2):
    """Indices where a new chunk starts, using a fixed similarity threshold."""
    boundaries = []
    centroid = None
    n = 0
    start = 0
    for i, s in enumerate(sentences):
        v = _normalized(embed, s)
        if centroid is None:
            centroid, n, start = v, 1, i
            continue
        if float(v @ centroid) < threshold and (i - start) >= min_size:
            boundaries.append(i)
            centroid, n, start = v, 1, i
        else:
            n += 1
            centroid = centroid + (v - centroid) / n
    return boundaries


def stream_boundaries_adaptive(sentences, embed, z=1.3, warmup=5, min_size=2):
    """Indices where a new chunk starts, using a self-calibrating threshold.

    Splits when a sentence's similarity to the running centroid falls more than
    ``z`` standard deviations below the mean similarity seen so far. Needs no
    absolute threshold; ``warmup`` sentences seed the running statistics.
    """
    boundaries = []
    centroid = None
    n = 0
    start = 0
    seen = []
    for i, s in enumerate(sentences):
        v = _normalized(embed, s)
        if centroid is None:
            centroid, n, start = v, 1, i
            continue
        sim = float(v @ centroid)
        drifted = len(seen) >= warmup and sim < (
            float(np.mean(seen)) - z * (float(np.std(seen)) + 1e-9)
        )
        if drifted and (i - start) >= min_size:
            boundaries.append(i)
            centroid, n, start = v, 1, i
        else:
            seen.append(sim)          # only in-topic similarities seed the baseline
            n += 1
            centroid = centroid + (v - centroid) / n
    return boundaries


def stream_chunks(
    sentences, embed, threshold=0.12, min_size=2, merge_short=True, min_words=4
):
    """Chunk ``sentences`` with a fixed threshold (single pass)."""
    if merge_short:
        sentences = merge_short_sentences(sentences, min_words)
    return _build_chunks(sentences, stream_boundaries(sentences, embed, threshold, min_size))


def stream_chunks_adaptive(
    sentences, embed, z=1.3, warmup=5, min_size=2, merge_short=True, min_words=4
):
    """Chunk ``sentences`` with a self-calibrating threshold (single pass)."""
    if merge_short:
        sentences = merge_short_sentences(sentences, min_words)
    bounds = stream_boundaries_adaptive(sentences, embed, z, warmup, min_size)
    return _build_chunks(sentences, bounds)
