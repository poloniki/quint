"""Single-pass, streaming semantic chunking.

Splits text into semantically coherent chunks in one forward pass — no O(n^2)
similarity matrix and no lookahead — so it works on a live stream. For each
sentence it compares the embedding to a running centroid of the current chunk
and opens a new chunk when the topic drifts away.

The split logic is embedding-agnostic: pass any callable mapping a sentence to a
vector (it is L2-normalized internally). See quint.chunking.generate.get_chunks
for the original matrix-based method.

Two guards keep the boundaries clean on real, noisy text:
- ``min_size`` forbids closing a chunk before it has that many sentences, which
  removes the spurious single-sentence chunks that short/odd sentences cause.
- ``merge_short`` folds very short sentences into their neighbour first, since
  short fragments have unreliable embeddings.
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


def stream_boundaries(sentences, embed, threshold=0.12, min_size=2):
    """Return the sentence indices where a new chunk starts (single pass).

    A boundary opens when a sentence's similarity to the running centroid drops
    below ``threshold`` — but only once the current chunk holds at least
    ``min_size`` sentences.
    """
    boundaries = []
    centroid = None
    n = 0
    start = 0
    for i, s in enumerate(sentences):
        v = np.asarray(embed(s), dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-9)
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


def stream_chunks(
    sentences, embed, threshold=0.12, min_size=2, merge_short=True, min_words=4
):
    """Chunk ``sentences`` into a list of paragraph strings (single pass)."""
    if merge_short:
        sentences = merge_short_sentences(sentences, min_words)
    bounds = stream_boundaries(sentences, embed, threshold, min_size)
    chunks, prev = [], 0
    for b in bounds + [len(sentences)]:
        chunk = " ".join(sentences[prev:b]).strip()
        if chunk:
            chunks.append(chunk)
        prev = b
    return chunks
