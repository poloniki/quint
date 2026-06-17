"""Quint — real-time semantic chunking demo (Hugging Face Space).

Watch paragraphs form *on the fly* from a transcript: static embeddings
(Model2Vec, no GPU) + a single-pass, self-calibrating streaming chunker. Mirrors
quint.chunking.streaming. Set OPENAI_API_KEY as a Space secret to enable the
summaries tab. Transcription (Whisper) needs a GPU — see the repo for the full API.
"""

import os
import time
from collections import deque

import gradio as gr
import numpy as np
import pysbd
from model2vec import StaticModel

HAS_KEY = bool(os.environ.get("OPENAI_API_KEY"))
seg = pysbd.Segmenter(language="en", clean=False)
model = StaticModel.from_pretrained("minishlab/potion-base-8M")


def embed(s):
    v = np.asarray(model.encode(s), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


SAMPLE = (
    "Welcome back to the show. Today we're talking about how small habits compound "
    "over time. A lot of people assume big results come from big, dramatic changes. "
    "In reality, most lasting change comes from tiny actions repeated daily. Reading "
    "ten pages a day feels trivial, but over a year that's more than a dozen books. "
    "The same logic applies to saving money or learning a language. The hard part isn't "
    "the size of the habit, it's the consistency. So how do you stay consistent? The "
    "first trick is to make the habit obvious. Put the book on your pillow, lay out your "
    "running shoes the night before. The second trick is to make it small enough that you "
    "can't say no. Two minutes is the magic number for most people. Another big factor is "
    "your environment. We like to think we rely on willpower, but our surroundings quietly "
    "shape most of our choices. If junk food is on the counter, you'll eat it. If it's in a "
    "cupboard you never open, you won't. Finally, track your streak. Seeing a chain of good "
    "days makes you reluctant to break it. So start tiny, stay consistent, and let the "
    "compounding do the work."
)

Z, WARMUP, MIN_SIZE, MIN_WORDS = 1.3, 5, 2, 4


def live_chunk(text, speed):
    """Generator: stream sentences in and yield the paragraphs forming live."""
    delay = {"fast": 0.04, "normal": 0.15, "slow": 0.4}.get(speed, 0.15)
    sents = seg.segment(text)
    done, cur, centroid, n = [], [], None, 0
    seen = deque(maxlen=256)

    def render(tail=""):
        md = ""
        for i, p in enumerate(done, 1):
            md += f"**¶ {i}**\n\n{p}\n\n---\n\n"
        if cur:
            md += f"*…forming ¶ {len(done) + 1}*\n\n> {' '.join(cur)}"
        return md + tail

    for s in sents:
        if cur and len(s.split()) < MIN_WORDS:          # online short-sentence merge
            cur[-1] = f"{cur[-1]} {s}"
            time.sleep(delay)
            yield render()
            continue
        v = embed(s)
        if centroid is None:
            cur, centroid, n = [s], v, 1
        else:
            sim = float(v @ centroid)
            drift = len(seen) >= WARMUP and sim < (
                float(np.mean(seen)) - Z * (float(np.std(seen)) + 1e-9)
            )
            if drift and len(cur) >= MIN_SIZE:
                done.append(" ".join(cur))
                cur, centroid, n = [s], v, 1
            else:
                seen.append(sim)
                cur.append(s)
                n += 1
                centroid = centroid + (v - centroid) / n
        time.sleep(delay)
        yield render()
    if cur:
        done.append(" ".join(cur))
    yield render(
        f"\n\n**✅ {len(done)} paragraphs — chunked in one streaming pass, "
        f"no fixed threshold (the cut point self-calibrates).**"
    )


def chunk_adaptive(sentences):
    cuts, centroid, n, start, seen = [], None, 0, 0, []
    for i, s in enumerate(sentences):
        v = embed(s)
        if centroid is None:
            centroid, n, start = v, 1, i
            continue
        sim = float(v @ centroid)
        drift = len(seen) >= WARMUP and sim < (
            float(np.mean(seen)) - Z * (float(np.std(seen)) + 1e-9)
        )
        if drift and (i - start) >= MIN_SIZE:
            cuts.append(i)
            centroid, n, start = v, 1, i
        else:
            seen.append(sim)
            n += 1
            centroid = centroid + (v - centroid) / n
    chunks, prev = [], 0
    for b in cuts + [len(sentences)]:
        chunk = " ".join(sentences[prev:b]).strip()
        if chunk:
            chunks.append(chunk)
        prev = b
    return chunks


def summarize(text):
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize in one sentence:\n\n{text}"}],
    )
    return resp.choices[0].message.content.strip()


def chunk_and_summarize(text):
    chunks = chunk_adaptive(seg.segment(text))
    head = f"**{len(chunks)} paragraphs**"
    if not HAS_KEY:
        head += "  \n_Set an `OPENAI_API_KEY` Space secret to enable summaries._"
    parts = [head, ""]
    for i, c in enumerate(chunks, 1):
        parts.append(f"### ¶ {i}")
        if HAS_KEY:
            try:
                parts.append(f"**Summary:** {summarize(c)}")
            except Exception as exc:
                parts.append(f"_(summary unavailable: {exc})_")
        parts.append(f"> {c}")
        parts.append("")
    return "\n".join(parts)


with gr.Blocks(title="Quint — real-time chunking") as demo:
    gr.Markdown(
        "# 🎙️ Quint — real-time semantic chunking\n"
        "Static embeddings + a single-pass, self-calibrating streaming chunker. "
        "[Code](https://github.com/poloniki/quint) · `pip install quintessentia`"
    )
    with gr.Tab("🔴 Live streaming"):
        gr.Markdown(
            "Watch paragraphs form **on the fly** as sentences arrive — one pass, "
            "no fixed threshold (the cut point self-calibrates from the stream)."
        )
        inp = gr.Textbox(value=SAMPLE, lines=8, label="Transcript (paste anything)")
        speed = gr.Radio(["fast", "normal", "slow"], value="normal", label="stream speed")
        btn = gr.Button("▶ Stream & chunk", variant="primary")
        out = gr.Markdown()
        btn.click(live_chunk, [inp, speed], out)
    with gr.Tab("📝 Chunk & summarize"):
        gr.Markdown("Split a transcript into paragraphs and summarize each.")
        inp2 = gr.Textbox(value=SAMPLE, lines=8, label="Transcript")
        btn2 = gr.Button("Chunk & summarize", variant="primary")
        out2 = gr.Markdown()
        btn2.click(chunk_and_summarize, inp2, out2)

if __name__ == "__main__":
    demo.launch()
