# Real-time, CPU-only Semantic Chunking — Blog Notebook (Part 3)

- **Date:** 2026-06-16
- **Status:** Approved design (pre-implementation)
- **Deliverable:** one self-contained Jupyter notebook,
  `notebooks/Chunking text into paragraphs V3 - real-time on CPU.ipynb`

## 1. Context & motivation

This is **Part 3** of the existing "chunking text into paragraphs" notebook series
that backs the project's Medium article and the `quint` package:

- **Part 1** (`Chunking text into paragraphs.ipynb`): pysbd sentences → MiniLM
  embeddings → full cosine-similarity matrix → rev_sigmoid-weighted diagonal sums
  → local minima = split points → matplotlib viz. This is the method in the package
  (`quint/chunking/`).
- **Part 2** (`…V2`, scratch): OpenAI embeddings → PCA → DBSCAN clustering.

Part 1's method has two hidden costs the post will dramatize:

1. A **~420 MB transformer** is required just to embed sentences (slow to load, wants a GPU to be quick).
2. The **O(n²) similarity matrix needs the entire document up front**, so it cannot
   operate on a live stream (e.g. Quint's incoming transcription).

## 2. Goal, audience, success criteria

**Goal:** teach one memorable idea — *you can chunk text semantically in real time,
on a CPU, in a single pass* — by replacing both costs above.

**Audience:** blog/portfolio readers (the Medium series). Optimize for the "aha"
and shareability, not production code.

**Success criteria:**
- The notebook runs **CPU-only in seconds**, no GPU.
- **Money shot #1 (benchmark):** a plot showing baseline (MiniLM + O(n²)) as a
  hockey-stick vs the new method (static embeddings + O(n) streaming) as a flat
  line, with a headline speedup number.
- **Money shot #2 (live demo):** sentences fed one at a time, chunk boundaries
  printed **as they form**, with a similarity-to-centroid trace that dips at each cut.
- A **quality check** showing the streaming boundaries are *comparable* to Part 1's
  (not garbage) — boundary agreement + a side-by-side excerpt.

## 3. The central idea (hook)

**Real-time CPU chunker = static embeddings + a streaming single-pass split rule.**

Two swaps vs Part 1:
- **Embeddings:** MiniLM (420 MB, GPU-friendly) → **Model2Vec static embeddings**
  (`minishlab/potion-base-8M`): distilled, no neural forward pass, microseconds per
  sentence, CPU-only.
- **Split decision:** global O(n²) similarity matrix → **running-centroid drift**,
  a single O(n) pass that never holds a matrix and needs no lookahead.

## 4. The algorithm (running-centroid drift)

Maintain the mean embedding of the *current* chunk. For each incoming sentence,
compare it to that centroid; if it has drifted past a threshold it doesn't belong,
so close the chunk and start a new one.

```python
def stream_chunks(sentences, embed, threshold=0.5):
    chunk, centroid, n = [], None, 0
    for s in sentences:
        v = embed(s)                          # static embedding — microseconds, no GPU
        if centroid is None:                  # first sentence starts a chunk
            chunk, centroid, n = [s], v, 1
            continue
        if cosine(v, centroid) < threshold:   # drifted away → boundary, on the fly
            yield " ".join(chunk)
            chunk, centroid, n = [s], v, 1
        else:                                 # belongs → extend + online-update mean
            chunk.append(s); n += 1
            centroid += (v - centroid) / n
    if chunk:
        yield " ".join(chunk)
```

- **Mental model:** *"is this sentence still about the same thing as the paragraph
  so far?"*
- **Properties:** single pass, O(n) time, O(1) extra state beyond the current chunk,
  one embedding per sentence, no matrix, no lookahead → works on a stream.
- **Normalization:** embeddings are L2-normalized (Model2Vec output already is; if not,
  normalize explicitly) so `cosine()` compares vectors on a comparable scale and the
  drift threshold has stable, model-agnostic semantics.
- **Refinement (featured in the caveats section):** a long chunk's centroid can drift
  away from its own start. The notebook **features the trailing-window centroid** (mean
  of the last *k* sentences instead of the whole chunk) as the fix, and **mentions** the
  relative-drop variant (similarity vs a short rolling baseline) as an alternative. Beat 6
  and the live-demo similarity trace use the trailing-window version for consistency.

## 5. Notebook structure (the six beats)

1. **Recap + the villain** — recap Part 1; name the two costs (420 MB model, O(n²)
   needs-whole-document).
2. **Fix #1 — ultra-fast embeddings** — load Model2Vec; show load size and per-sentence
   embed time (microseconds); a 3-line cosine sanity check that topics still separate
   (e.g. two on-topic sentences vs an off-topic one).
3. **Fix #2 — the streaming chunker** — implement `stream_chunks`; explain the threshold
   and the online mean update; run it on the transcript.
4. **Money shot #1 — benchmark** — baseline vs new, wall-clock vs document length
   (≈50 → 10 000 sentences, transcript concatenated to scale); plot both curves;
   report the speedup. Measure embed time and split time; report model **load** time
   separately so the comparison is fair.
5. **Money shot #2 — watch it live** — a generator yields sentences with a tiny sleep
   to simulate Quint's live transcription; print each chunk as it closes; plot
   similarity-to-centroid across the stream with vertical lines at the cuts.
6. **Caveats & where next** — threshold selection (show the similarity distribution),
   centroid drift + the trailing-window fix, pointers to the adjacent-similarity and
   rolling-percentile variants, and multilingual static models.

## 6. Tech & data

- **Embeddings:** `model2vec` (`minishlab/potion-base-8M`).
- **Sentence splitting:** `pysbd` (consistent with Parts 1–2).
- **Baseline reuse:** the Part-1 method (MiniLM + cosine matrix + activated-similarity
  minima) for the benchmark/quality comparison.
- **Math/plots:** `numpy`, `matplotlib`.
- **Data:** `data/podcast_transcript.txt` (on-brand for Quint), concatenated to scale
  up for the benchmark.

## 7. Risks & open questions

- **Static-embedding quality:** Model2Vec is lossy vs MiniLM. The boundaries must look
  *reasonable*, not necessarily identical. **Fallback** if the demo looks poor: distill
  MiniLM into a static model via `model2vec.distill`, or fall back to a small
  sentence-transformer (still CPU-fast) — speed story holds either way.
- **Threshold tuning:** the cosine threshold is model-dependent. Pick it by plotting the
  similarity distribution on the transcript and choosing a sensible cut; expose it as a
  parameter rather than hard-coding a magic number silently.
- **Benchmark fairness:** warm up both models before timing; use the same scaled text;
  report load time separately; average over a few runs with `time.perf_counter`.
- **Quality metric:** "comparable to Part 1" = boundary agreement within ±1 sentence
  plus a side-by-side excerpt — a sanity check, not a rigorous eval. **Run the quality
  comparison on the original (un-concatenated) transcript**: concatenated text is highly
  self-similar and would make boundaries look artificially clean. Concatenation is used
  *only* for the speed benchmark (beat 4), where the point is wall-clock vs length.

## 8. Out of scope (YAGNI)

- No changes to the `quint` package (the notebook *links* to it).
- No packaging, no API, no UI, no LLM/agentic chunking.
- Not a rigorous evaluation — it is a teaching artifact.
