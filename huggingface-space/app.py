"""Quint — real-time speech → paragraphs demo (Hugging Face Space).

A full little product: speak into the mic, watch it transcribe live (sherpa-onnx
streaming CPU speech-to-text) and group the utterances into semantic paragraphs
on the fly (Model2Vec static embeddings + a single-pass, self-calibrating chunker,
mirroring quint.chunking.streaming). No GPU. Set OPENAI_API_KEY as a Space secret
for the summaries tab.
"""

import glob
import os
import tarfile
import time
import urllib.request
from collections import deque

import gradio as gr
import numpy as np
import pysbd
from model2vec import StaticModel

seg = pysbd.Segmenter(language="en", clean=False)
model = StaticModel.from_pretrained("minishlab/potion-base-8M")
Z, WARMUP, MIN_SIZE, MIN_WORDS = 1.3, 5, 2, 4


def embed(s):
    v = np.asarray(model.encode(s), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ------------------------------------------------------------------ speech-to-text
RECOGNIZER = None
_STT_MODEL = "sherpa-onnx-streaming-zipformer-en-2023-06-26"


def _load_stt():
    """Best-effort: download the small streaming model and build the recognizer."""
    global RECOGNIZER
    try:
        import sherpa_onnx

        if not os.path.isdir(_STT_MODEL):
            url = (
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                f"asr-models/{_STT_MODEL}.tar.bz2"
            )
            print("downloading STT model…")
            urllib.request.urlretrieve(url, "stt.tar.bz2")
            with tarfile.open("stt.tar.bz2") as t:
                t.extractall(".")
        enc = glob.glob(f"{_STT_MODEL}/encoder*int8.onnx")[0]
        dec = glob.glob(f"{_STT_MODEL}/decoder*.onnx")[0]
        joi = glob.glob(f"{_STT_MODEL}/joiner*int8.onnx")[0]
        RECOGNIZER = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{_STT_MODEL}/tokens.txt", encoder=enc, decoder=dec, joiner=joi,
            num_threads=2, sample_rate=16000, feature_dim=80,
            enable_endpoint_detection=True, decoding_method="greedy_search",
        )
        print("STT ready")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully if STT can't load
        print("STT unavailable:", exc)


_load_stt()

# one global session (this is a single-user demo)
_S = {}


def _reset():
    _S.clear()
    _S.update(stream=None, transcript="", paras=[], cur=[], centroid=None, n=0,
              seen=deque(maxlen=256))


_reset()


def _feed(utterance):
    """Online-chunk one finalized utterance (treated as one 'sentence')."""
    v = embed(utterance)
    if _S["centroid"] is None:
        _S["cur"], _S["centroid"], _S["n"] = [utterance], v, 1
        return
    sim = float(v @ _S["centroid"])
    drift = len(_S["seen"]) >= WARMUP and sim < (
        float(np.mean(_S["seen"])) - Z * (float(np.std(_S["seen"])) + 1e-9)
    )
    if drift and len(_S["cur"]) >= MIN_SIZE:
        _S["paras"].append(" ".join(_S["cur"]))
        _S["cur"], _S["centroid"], _S["n"] = [utterance], v, 1
    else:
        _S["seen"].append(sim)
        _S["cur"].append(utterance)
        _S["n"] += 1
        _S["centroid"] = _S["centroid"] + (v - _S["centroid"]) / _S["n"]


def _render(partial=""):
    live = (_S["transcript"] + " " + partial).strip()
    md = f"#### 🎧 live transcript\n> {live or '…'}\n\n#### 📑 paragraphs\n"
    paras = _S["paras"] + ([" ".join(_S["cur"])] if _S["cur"] else [])
    for i, p in enumerate(paras, 1):
        md += f"**¶ {i}** — {p}\n\n"
    return md


def on_mic(new_chunk):
    if RECOGNIZER is None:
        return "Speech-to-text model is still loading or unavailable — try the other tabs."
    if new_chunk is None:
        return _render()
    sr, y = new_chunk
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.max(np.abs(y)) > 1.5:      # int16 -> float [-1, 1]
        y = y / 32768.0
    if _S["stream"] is None:
        _S["stream"] = RECOGNIZER.create_stream()
    st = _S["stream"]
    st.accept_waveform(sr, y)
    while RECOGNIZER.is_ready(st):
        RECOGNIZER.decode_stream(st)
    partial = RECOGNIZER.get_result(st).strip().lower()
    if RECOGNIZER.is_endpoint(st):
        if partial:
            _S["transcript"] += " " + partial
            _feed(partial)
        RECOGNIZER.reset(st)
        partial = ""
    return _render(partial)


# ------------------------------------------------------------------ text streaming
SAMPLE = (
    "Welcome back to the show. Today we're talking about how small habits compound "
    "over time. A lot of people assume big results come from big, dramatic changes. "
    "In reality, most lasting change comes from tiny actions repeated daily. Reading "
    "ten pages a day feels trivial, but over a year that's more than a dozen books. "
    "The same logic applies to saving money or learning a language. The hard part isn't "
    "the size of the habit, it's the consistency. So how do you stay consistent? The "
    "first trick is to make the habit obvious. The second trick is to make it small "
    "enough that you can't say no. Two minutes is the magic number for most people. "
    "Another big factor is your environment. We like to think we rely on willpower, but "
    "our surroundings quietly shape most of our choices. If junk food is on the counter, "
    "you'll eat it. Finally, track your streak. Seeing a chain of good days makes you "
    "reluctant to break it. So start tiny, stay consistent, and let the compounding do "
    "the work."
)


def live_chunk(text, speed):
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
        if cur and len(s.split()) < MIN_WORDS:
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
    yield render(f"\n\n**✅ {len(done)} paragraphs — one streaming pass, no fixed threshold.**")


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


def chunk_and_summarize(text):
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    chunks = chunk_adaptive(seg.segment(text))
    head = f"**{len(chunks)} paragraphs**"
    if not has_key:
        head += "  \n_Set an `OPENAI_API_KEY` Space secret to enable summaries._"
    parts = [head, ""]
    for i, c in enumerate(chunks, 1):
        parts.append(f"### ¶ {i}")
        if has_key:
            try:
                from openai import OpenAI

                resp = OpenAI().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"Summarize in one sentence:\n\n{c}"}],
                )
                parts.append(f"**Summary:** {resp.choices[0].message.content.strip()}")
            except Exception as exc:  # noqa: BLE001
                parts.append(f"_(summary unavailable: {exc})_")
        parts.append(f"> {c}")
        parts.append("")
    return "\n".join(parts)


# ------------------------------------------------------------------ UI
with gr.Blocks(title="Quint — real-time speech to paragraphs") as demo:
    gr.Markdown(
        "# 🎙️ Quint — real-time speech → paragraphs\n"
        "Speak (or paste text) and watch it transcribe and group into semantic "
        "paragraphs **live, on a CPU**. [Code](https://github.com/poloniki/quint) · "
        "`pip install quintessentia`"
    )
    with gr.Tab("🎙️ Live mic"):
        gr.Markdown(
            "Click the mic, allow access, and start talking. It transcribes with a "
            "small streaming model and groups your sentences into paragraphs on the fly. "
            "_English; first use downloads a ~70 MB model._"
        )
        mic = gr.Audio(sources=["microphone"], streaming=True, label="microphone")
        clear = gr.Button("Clear / restart")
        mic_out = gr.Markdown()
        mic.stream(on_mic, mic, mic_out)
        clear.click(lambda: (_reset(), _render())[1], None, mic_out)
    with gr.Tab("🔴 Live streaming (text)"):
        gr.Markdown("No mic? Paste a transcript and watch the paragraphs form, one pass.")
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
