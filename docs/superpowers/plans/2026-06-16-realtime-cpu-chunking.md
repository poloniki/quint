# Real-time CPU Semantic Chunking Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/Chunking text into paragraphs V3 - real-time on CPU.ipynb` — a Part-3 blog notebook teaching real-time, CPU-only semantic chunking — with **real, executed benchmark numbers and plots**, not placeholders.

**Architecture:** One self-contained notebook. Swap Part 1's MiniLM (420 MB) for Model2Vec static embeddings, and its O(n²) similarity matrix for a single-pass running-centroid streaming chunker. Build it cell-group by cell-group; **execute every code cell for real** (`jupyter nbconvert --execute`) so the committed notebook contains true timings/plots. Reuse the Part-1 method from `quint/chunking/` as the benchmark baseline.

**Tech Stack:** Python 3.10, `model2vec`, `pysbd`, `sentence-transformers` (baseline), `numpy`, `matplotlib`, `jupyter`/`nbconvert`. CPU only.

**Spec:** `docs/superpowers/specs/2026-06-16-realtime-cpu-chunking-design.md`

---

## Conventions for every task

- The notebook is built/edited as a real `.ipynb`. Add cells with the `NotebookEdit` tool (or a small JSON builder script), keeping the existing series' `python3` kernelspec.
- **"Verify" always means execute for real**, never hand-write an output. Use:
  `jupyter nbconvert --to notebook --execute --inplace "notebooks/Chunking text into paragraphs V3 - real-time on CPU.ipynb"`
  then confirm it ran without errors and inspect the captured outputs.
- Commit after each task. Run from repo root `/Users/poloniki/code/poloniki/quint`.
- The `~/.pyenv/versions/quint` env is the kernel; `sentence-transformers`, `pysbd`, `numpy`, `matplotlib` are already installed there.

---

## File Structure

- **Create:** `notebooks/Chunking text into paragraphs V3 - real-time on CPU.ipynb` — the whole deliverable.
- **Read-only (baseline reuse):** `quint/chunking/generate.py` (`get_chunks`), `quint/tools/embedding.py` (`create_embedding`), `quint/chunking/similarities.py`.
- **Read-only (data):** `data/podcast_transcript.txt`.
- No package or test files are modified — this is a teaching artifact (spec §8).

---

### Task 1: Environment + notebook scaffold

**Files:** Create `notebooks/Chunking text into paragraphs V3 - real-time on CPU.ipynb`

- [ ] **Step 1: Install/verify deps in the quint env**

Run:
```bash
python3 -m pip install --quiet model2vec jupyter nbconvert ipykernel matplotlib
python3 -c "from model2vec import StaticModel; m=StaticModel.from_pretrained('minishlab/potion-base-8M'); import numpy as np; print('dim', np.asarray(m.encode('hello world')).shape)"
```
Expected: prints a dim like `dim (256,)` (downloads the ~30 MB static model once). If `model2vec` fails to install, STOP and use the spec §7 fallback (`model2vec.distill` from MiniLM, or a small sentence-transformer) — record which path was taken.

- [ ] **Step 2: Create the notebook with a title + intro markdown cell and an imports code cell**

Title markdown (cell 0):
```markdown
# Chunking text into paragraphs — Part 3: real-time, on a CPU

In Part 1 we split text by building a full sentence-similarity **matrix** and finding its
valleys. It works — but it needs a 420 MB model and the **whole document up front**, so it
can't keep up with a live stream. This part fixes both: **static embeddings** (microseconds,
no GPU) + a **single-pass streaming** splitter. Same idea, real-time, on a laptop.
```
Imports (cell 1):
```python
import time
import numpy as np
import matplotlib.pyplot as plt
import pysbd
from model2vec import StaticModel
```

- [ ] **Step 3: Execute the notebook for real**

Run the `nbconvert --execute` command from Conventions. Expected: exit 0, imports cell has no error output.

- [ ] **Step 4: Commit**

```bash
git add "notebooks/Chunking text into paragraphs V3 - real-time on CPU.ipynb"
git commit -m "feat(nb): scaffold Part-3 real-time chunking notebook"
```

---

### Task 2: Beat 1 — load the transcript and split into sentences

**Files:** Modify the notebook (add cells).

- [ ] **Step 1: Add a markdown "Recap + the two costs" cell** summarizing Part 1 and naming the two villains (420 MB model; O(n²) matrix needs the whole document).

- [ ] **Step 2: Add a code cell** that loads the data and segments sentences:
```python
text = open("../data/podcast_transcript.txt").read()
seg = pysbd.Segmenter(language="en", clean=False)
sentences = seg.segment(text)
len(sentences), sentences[0][:80]
```

- [ ] **Step 3: Execute for real**, confirm it prints a real sentence count (> 0) and a sample.

- [ ] **Step 4: Commit** `feat(nb): beat 1 — load transcript + sentences`.

---

### Task 3: Beat 2 — ultra-fast static embeddings (Fix #1)

**Files:** Modify the notebook.

- [ ] **Step 1: Markdown** — "Fix #1: stop using a 420 MB transformer."

- [ ] **Step 2: Code cell** — load the static model and time per-sentence embedding:
```python
static = StaticModel.from_pretrained("minishlab/potion-base-8M")

def embed(s):
    v = np.asarray(static.encode(s), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)   # L2-normalize so cosine == dot

t0 = time.perf_counter()
_ = [embed(s) for s in sentences]
per = (time.perf_counter() - t0) / len(sentences)
print(f"{per*1e6:.1f} µs/sentence, CPU only, model on disk ~30 MB")
```

- [ ] **Step 3: Code cell — 3-line sanity check** that topics still separate:
```python
def cos(a, b): return float(a @ b)        # inputs are normalized
a, b, c = embed("The cat sat on the mat."), embed("My dog chased the ball."), embed("Q3 revenue beat estimates.")
print("pet~pet", round(cos(a, b), 3), " pet~finance", round(cos(a, c), 3))
```
Expected at execution: pet~pet noticeably > pet~finance. If not (static model too lossy), switch to the spec §7 fallback and note it in a markdown cell.

- [ ] **Step 4: Execute for real**; confirm a real µs/sentence number prints and the sanity check separates topics.

- [ ] **Step 5: Commit** `feat(nb): beat 2 — Model2Vec static embeddings + timing`.

---

### Task 4: Beat 3 — the streaming chunker (Fix #2, the core)

**Files:** Modify the notebook.

- [ ] **Step 1: Code cell — the algorithm** (the teachable core):
```python
def stream_chunks(sentences, embed, threshold=0.5):
    chunk, centroid, n = [], None, 0
    for s in sentences:
        v = embed(s)                              # static embedding — microseconds
        if centroid is None:
            chunk, centroid, n = [s], v, 1
            continue
        if float(v @ centroid) < threshold:       # drifted away → boundary, on the fly
            yield " ".join(chunk)
            chunk, centroid, n = [s], v, 1
        else:
            chunk.append(s); n += 1
            centroid = centroid + (v - centroid) / n
    if chunk:
        yield " ".join(chunk)
```

- [ ] **Step 2: Code cell — correctness assert (TDD)** on a toy input with an obvious boundary:
```python
toy = ["I love training neural networks.", "Backprop tunes the weights.",
       "Yesterday I baked sourdough bread.", "The crust came out crispy."]
out = list(stream_chunks(toy, embed, threshold=0.5))
assert len(out) == 2, out          # ML sentences vs baking sentences -> 2 chunks
print(out)
```
If the assert fails, tune `threshold` (Task 7 shows how to pick it) and record the chosen value.

- [ ] **Step 3: Code cell** — run on the real transcript and show the first few chunks + count.

- [ ] **Step 4: Execute for real**; confirm the assert passes and real chunks print.

- [ ] **Step 5: Commit** `feat(nb): beat 3 — single-pass running-centroid chunker`.

---

### Task 5: Beat 4 — Money shot #1: the REAL benchmark

**Files:** Modify the notebook. **This task must produce real measured numbers.**

- [ ] **Step 1: Markdown** — "How fast, really? Old vs new as the document grows."

- [ ] **Step 2: Code cell — define both pipelines and the scaling harness:**
```python
from quint.chunking.generate import get_chunks    # Part-1 baseline: MiniLM + O(n^2) matrix

def new_pipeline(text):
    sents = seg.segment(text)
    return list(stream_chunks(sents, embed, threshold=0.5))

def timed(fn, text, repeats=3):
    fn(text)                                        # warm up
    best = min(_t(fn, text) for _ in range(repeats))
    return best

def _t(fn, text):
    t0 = time.perf_counter(); fn(text); return time.perf_counter() - t0

base_text = open("../data/podcast_transcript.txt").read()
def scale_to(n_sentences):
    s = seg.segment(base_text)
    reps = (n_sentences // len(s)) + 1
    return " ".join((s * reps)[:n_sentences])
```

- [ ] **Step 3: Code cell — run the benchmark for real.** The O(n²) baseline hits a memory/time wall, so cap it; let the new method run far past it (this divergence IS the story):
```python
sizes = [50, 100, 250, 500, 1000, 2000, 4000, 8000]
base_cap = 2000                                     # baseline's feasible ceiling (n^2 matrix)
rows = []
for n in sizes:
    txt = scale_to(n)
    t_new = timed(new_pipeline, txt)
    t_base = timed(get_chunks, txt) if n <= base_cap else None
    rows.append((n, t_base, t_new))
    print(f"n={n:5d}  baseline={'%.3fs'%t_base if t_base else '  —  (O(n^2) wall)':>14}  new={t_new:.4f}s")
```

- [ ] **Step 4: Code cell — plot the two curves + headline number:**
```python
ns   = [r[0] for r in rows]
tnew = [r[2] for r in rows]
nb_x = [r[0] for r in rows if r[1] is not None]
nb_y = [r[1] for r in rows if r[1] is not None]
plt.figure(figsize=(7,4))
plt.plot(nb_x, nb_y, "o-", label="Part 1: MiniLM + O(n²) matrix")
plt.plot(ns,   tnew, "o-", label="Part 3: static + O(n) streaming")
plt.xlabel("sentences"); plt.ylabel("seconds"); plt.legend(); plt.title("Chunking time vs document length")
plt.show()
# headline at the largest size both ran
both = [r for r in rows if r[1] is not None]
n0, tb, tn = both[-1]
print(f"At n={n0}: {tb:.3f}s  ->  {tn:.4f}s  =  {tb/tn:.0f}x faster (and the new one keeps going past the matrix's memory wall)")
```

- [ ] **Step 5: Code cell — quality check on the ORIGINAL (un-concatenated) transcript** (spec: concatenated text is self-similar, so judge quality on the real transcript):
```python
base_chunks = get_chunks(base_text)
new_chunks  = new_pipeline(base_text)
print(f"baseline made {len(base_chunks)} chunks, streaming made {len(new_chunks)}")

def cut_indices(chunks):                       # sentence index of each chunk boundary
    cuts, acc = set(), 0
    for c in chunks[:-1]:
        acc += len(seg.segment(c)); cuts.add(acc)
    return cuts
bA, nA = cut_indices(base_chunks), cut_indices(new_chunks)
agree = sum(any(abs(b - x) <= 1 for x in bA) for b in nA) / max(len(nA), 1)
print(f"{agree:.0%} of streaming cuts land within ±1 sentence of a Part-1 cut")
print("first streaming chunk:\n", new_chunks[0][:300])
```

- [ ] **Step 6: Execute for real.** Confirm: real per-size timings printed, the speedup is a real measured number (> 1x), the plot rendered, and the baseline shows the expected quadratic growth. Record the headline speedup in the markdown.

- [ ] **Step 7: Commit** `feat(nb): beat 4 — real speed benchmark + quality check`.

---

### Task 6: Beat 5 — Money shot #2: watch it chunk live

**Files:** Modify the notebook.

- [ ] **Step 1: Markdown** — "Because it's single-pass, it works on a live stream (like Quint's transcription)."

- [ ] **Step 2: Code cell — streaming demo that prints boundaries as they form and records the similarity trace:**
```python
def stream_with_trace(sentences, embed, threshold=0.5, delay=0.0):
    chunk, centroid, n, sims, cuts = [], None, 0, [], []
    for i, s in enumerate(sentences):
        time.sleep(delay)                            # simulate live arrival
        v = embed(s)
        sim = 1.0 if centroid is None else float(v @ centroid)
        sims.append(sim)
        if centroid is not None and sim < threshold:
            cuts.append(i); print("── new paragraph ──")
            chunk, centroid, n = [s], v, 1
        else:
            if centroid is None: chunk, centroid, n = [s], v, 1
            else: chunk.append(s); n += 1; centroid = centroid + (v - centroid)/n
        print(f"  [{sim:.2f}] {s[:70]}")
    return sims, cuts

sims, cuts = stream_with_trace(sentences[:60], embed, threshold=0.5, delay=0.0)
```

- [ ] **Step 3: Code cell — plot the similarity-to-centroid trace with the cuts marked:**
```python
plt.figure(figsize=(9,3))
plt.plot(sims, lw=1)
for c in cuts: plt.axvline(c, color="r", ls="--", alpha=.5)
plt.axhline(0.5, color="k", ls=":", alpha=.4)
plt.xlabel("sentence # (stream order)"); plt.ylabel("cos to current centroid"); plt.title("Similarity dips → cuts")
plt.show()
```

- [ ] **Step 4: Execute for real**; confirm the incremental prints, the cuts, and the trace plot all render.

- [ ] **Step 5: Commit** `feat(nb): beat 5 — live streaming demo + similarity trace`.

---

### Task 7: Beat 6 — caveats, threshold picking, and the drift fix

**Files:** Modify the notebook.

- [ ] **Step 1: Code cell — how to pick the threshold** (plot the distribution of sentence-to-centroid similarities so the cut point isn't a magic number):
```python
def all_sims(sentences, embed):
    centroid, n, out = None, 0, []
    for s in sentences:
        v = embed(s)
        if centroid is None: centroid, n = v, 1; continue
        out.append(float(v @ centroid)); centroid = centroid + (v - centroid)/(n+1); n += 1
    return out
plt.hist(all_sims(sentences, embed), bins=40); plt.title("pick the threshold in the valley"); plt.show()
```

- [ ] **Step 2: Code cell — the featured drift fix: trailing-window centroid** (mean of the last *k* sentences instead of the whole chunk):
```python
def stream_chunks_window(sentences, embed, threshold=0.5, k=5):
    chunk, vecs = [], []
    for s in sentences:
        v = embed(s)
        if not chunk:
            chunk, vecs = [s], [v]; continue
        centroid = np.mean(vecs[-k:], axis=0); centroid /= (np.linalg.norm(centroid)+1e-9)
        if float(v @ centroid) < threshold:
            yield " ".join(chunk); chunk, vecs = [s], [v]
        else:
            chunk.append(s); vecs.append(v)
    if chunk: yield " ".join(chunk)
print(len(list(stream_chunks_window(sentences, embed))), "chunks (trailing-window variant)")
```

- [ ] **Step 3: Markdown caveats** — centroid drift (and why the trailing window helps), the relative-drop alternative (mention only), threshold tuning, and pointers: adjacent-similarity & rolling-percentile variants, multilingual static models. End with a link back to `quint` / `pip install quintessentia`.

- [ ] **Step 4: Execute for real**; confirm both plots/numbers render.

- [ ] **Step 5: Commit** `feat(nb): beat 6 — caveats, threshold picking, drift fix`.

---

### Task 8: Final end-to-end execution + ship

**Files:** the notebook.

- [ ] **Step 1: Execute the WHOLE notebook top-to-bottom, clean:**
```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
  "notebooks/Chunking text into paragraphs V3 - real-time on CPU.ipynb"
```
Expected: exit 0, no cell errors.

- [ ] **Step 2: Verify the committed notebook contains REAL outputs** (not empty): confirm the benchmark cell's output has measured seconds, the speedup line has a real number, and the three plots have image output. Quick check:
```bash
python3 -c "import json; nb=json.load(open('notebooks/Chunking text into paragraphs V3 - real-time on CPU.ipynb')); outs=sum(len(c.get('outputs',[])) for c in nb['cells'] if c['cell_type']=='code'); imgs=sum('image/png' in (o.get('data',{}) ) for c in nb['cells'] for o in c.get('outputs',[])); print('code outputs:', outs, '| image outputs:', imgs)"
```
Expected: `code outputs` > 0 and `image outputs` >= 3.

- [ ] **Step 3: Final commit**
```bash
git add "notebooks/Chunking text into paragraphs V3 - real-time on CPU.ipynb"
git commit -m "feat(nb): Part-3 real-time CPU chunking notebook with executed benchmarks"
```

- [ ] **Step 4 (optional): push** if the user wants it published.

---

## Definition of done

- Notebook runs top-to-bottom on CPU with no errors via `nbconvert --execute`.
- The benchmark shows **real measured** timings: baseline grows quadratically (and hits a wall), new method stays ~flat; a real speedup number is printed.
- The live demo prints boundaries forming and renders the similarity trace.
- The quality check (on the un-concatenated transcript) shows comparable boundaries.
- Scope held: one notebook, no package changes.
