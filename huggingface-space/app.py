"""Quint — real-time speech → paragraphs demo (Hugging Face Space).

Speak into the mic and watch it transcribe live (sherpa-onnx streaming CPU
speech-to-text), restore punctuation (sherpa-onnx CT-transformer), and group the
sentences into semantic paragraphs on the fly (Model2Vec static embeddings + a
single-pass, self-calibrating chunker, mirroring quint.chunking.streaming). No
GPU. Set OPENAI_API_KEY as a Space secret for the summaries tab.
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


# ------------------------------------------------------------------ speech models
RECOGNIZER = None
PUNCT = None
_STT_MODEL = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
_PUNCT_MODEL = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"


def _fetch(name, kind):
    if not os.path.isdir(name):
        url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/{kind}/{name}.tar.bz2"
        print(f"downloading {name}…")
        urllib.request.urlretrieve(url, "m.tar.bz2")
        with tarfile.open("m.tar.bz2") as t:
            t.extractall(".")


def _load_models():
    global RECOGNIZER, PUNCT
    try:
        import sherpa_onnx

        _fetch(_STT_MODEL, "asr-models")
        enc = glob.glob(f"{_STT_MODEL}/encoder*int8.onnx")[0]
        dec = glob.glob(f"{_STT_MODEL}/decoder*.onnx")[0]
        joi = glob.glob(f"{_STT_MODEL}/joiner*int8.onnx")[0]
        RECOGNIZER = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{_STT_MODEL}/tokens.txt", encoder=enc, decoder=dec, joiner=joi,
            num_threads=2, sample_rate=16000, feature_dim=80,
            enable_endpoint_detection=True, decoding_method="greedy_search",
        )
        print("STT ready")
        _fetch(_PUNCT_MODEL, "punctuation-models")
        PUNCT = sherpa_onnx.OfflinePunctuation(
            sherpa_onnx.OfflinePunctuationConfig(
                model=sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=f"{_PUNCT_MODEL}/model.onnx", num_threads=1
                )
            )
        )
        print("punctuation ready")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        print("speech models unavailable:", exc)


_load_models()

_S = {}


def _reset():
    _S.clear()
    _S.update(stream=None, raw="", paras=[], display="", since=0)


_reset()

_CJK = {"。": ". ", "，": ", ", "？": "? ", "！": "! ", "、": ", ", "；": "; ", "：": ": "}


def _punctuate(text):
    """Restore punctuation on raw ASR text, normalize to ASCII, capitalize starts."""
    text = text.strip()
    if not text:
        return ""
    if PUNCT is not None:
        text = PUNCT.add_punctuation(text)
        for a, b in _CJK.items():
            text = text.replace(a, b)
    out, cap = [], True
    for ch in text:
        out.append(ch.upper() if cap and ch.isalpha() else ch)
        if ch in ".!?":
            cap = True
        elif not ch.isspace():
            cap = False
    return "".join(out)


def _recompute(text):
    """Re-punctuate, re-segment, and re-chunk the running transcript."""
    punctuated = _punctuate(text)
    sents = seg.segment(punctuated) if punctuated else []
    if len(sents) > 1:
        _S["paras"] = chunk_adaptive(sents)
    else:
        _S["paras"] = [punctuated] if punctuated else []
    body = "".join(f"**¶ {i}** — {p}\n\n" for i, p in enumerate(_S["paras"], 1))
    _S["display"] = f"#### 📑 paragraphs\n\n{body or '_listening…_'}"


def on_mic(new_chunk):
    if RECOGNIZER is None:
        return "Speech model still loading (first run downloads ~360 MB) or unavailable — try the text tabs."
    if new_chunk is None:
        return _S["display"] or "_listening…_"
    sr, y = new_chunk
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if np.max(np.abs(y)) > 1.5:                 # int16 -> float [-1, 1]
        y = y / 32768.0
    if _S["stream"] is None:
        _S["stream"] = RECOGNIZER.create_stream()
    st = _S["stream"]
    st.accept_waveform(sr, y)
    while RECOGNIZER.is_ready(st):
        RECOGNIZER.decode_stream(st)
    partial = RECOGNIZER.get_result(st).strip().lower()
    endpoint = RECOGNIZER.is_endpoint(st)
    if endpoint:
        if partial:
            _S["raw"] = f"{_S['raw']} {partial}".strip()
        RECOGNIZER.reset(st)
        partial = ""
    _S["since"] += 1
    if endpoint or _S["since"] >= 10:            # throttle re-punctuation
        _S["since"] = 0
        _recompute(f"{_S['raw']} {partial}".strip())
    return _S["display"] or "_listening…_"


def _clear_mic():
    _reset()
    return "Reset — start talking…"


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


def merge_short(sentences, min_words=MIN_WORDS):
    """Fold sub-``min_words`` fragments (common in raw ASR) into the previous one."""
    out = []
    for s in sentences:
        if out and len(s.split()) < min_words:
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out


def chunk_adaptive(sentences):
    sentences = merge_short(sentences)
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
        "Speak (or paste text) and watch it transcribe, punctuate, and group into "
        "semantic paragraphs **live, on a CPU**. "
        "[Code](https://github.com/poloniki/quint) · `pip install quintessentia`"
    )
    with gr.Tab("🎙️ Live mic"):
        gr.Markdown(
            "Click the mic, allow access, and start talking. It transcribes with a "
            "small streaming model, restores punctuation, and groups your sentences "
            "into paragraphs on the fly. _English; first use downloads ~360 MB of models, "
            "so the very first response can take a minute._"
        )
        mic = gr.Audio(sources=["microphone"], streaming=True, label="microphone")
        clear = gr.Button("Clear / restart")
        mic_out = gr.Markdown()
        mic.stream(on_mic, mic, mic_out)
        clear.click(_clear_mic, None, mic_out)
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
