"""Quint — real-time speech → paragraphs demo (Hugging Face Space).

Speak into the mic and watch it transcribe and group into semantic paragraphs.
Speech-to-text is Moonshine (sherpa-onnx, ONNX/CPU, no torch) — a fast model
whose compute scales with the actual audio length (unlike Whisper's fixed 30s
window), so short pause-delimited utterances transcribe in tens of milliseconds.
We buffer the mic audio, detect a pause, transcribe the utterance — Moonshine
returns clean, punctuated, cased text — then segment it and group the sentences
into semantic paragraphs with Model2Vec static embeddings + a single-pass,
self-calibrating chunker (mirroring quint.chunking.streaming). No GPU.
Set OPENAI_API_KEY as a Space secret for the summaries tab.
"""

import glob
import os
import re
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
Z, WARMUP, MIN_SIZE, MIN_WORDS = 1.3, 3, 2, 4   # warmup 3 (vs lib's 5): short demo inputs
MAX_PARA_SENTS = 6                              # readability cap: break very long paragraphs


def embed(s):
    v = np.asarray(model.encode(s), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


# ------------------------------------------------------------------ speech-to-text
RECOGNIZER = None
_ASR_MODEL = "sherpa-onnx-moonshine-base-en-int8"   # ~270 MB int8, punctuated + cased


def _fetch(name, kind):
    if not os.path.isdir(name):
        url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/{kind}/{name}.tar.bz2"
        print(f"downloading {name}…")
        urllib.request.urlretrieve(url, "m.tar.bz2")
        with tarfile.open("m.tar.bz2") as t:
            t.extractall(".")


def _load_models():
    global RECOGNIZER
    try:
        import sherpa_onnx

        _fetch(_ASR_MODEL, "asr-models")
        RECOGNIZER = sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=glob.glob(f"{_ASR_MODEL}/preprocess*.onnx")[0],
            encoder=glob.glob(f"{_ASR_MODEL}/encode*.onnx")[0],
            uncached_decoder=glob.glob(f"{_ASR_MODEL}/uncached_decode*.onnx")[0],
            cached_decoder=glob.glob(f"{_ASR_MODEL}/cached_decode*.onnx")[0],
            tokens=glob.glob(f"{_ASR_MODEL}/tokens.txt")[0],
            num_threads=2,
        )
        print("moonshine ready")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully if unavailable
        print("speech model unavailable:", exc)


_load_models()

# ---- mic session: buffer audio, flush an utterance on a pause. Per-session state
# lives in gr.State (NOT a module global) so concurrent viewers don't collide.
SR = 16000
SILENCE_RMS = 0.008          # below this a chunk counts as silence
PAUSE_SEC = 0.5             # this much quiet after speech closes an utterance
MIN_UTT = SR * 1            # need >= 1 s of audio before transcribing
MAX_UTT = SR * 15          # safety flush so a runaway buffer can't stall
# Moonshine/Whisper invent stock filler on silence/noise — drop such whole sentences
_HALLUC_SENTS = {
    "you", "thank you", "thank you very much", "thank you so much", "thanks",
    "thanks for watching", "thank you for watching", "thank you my lord",
    "please subscribe", "bye", "bye bye", "goodbye", "yeah", "yep", "okay",
    "ok", "mm", "mm-hmm", "i'm sorry", "sorry",
}
_HALLUC_RE = re.compile(r"[^a-z'\- ]")


def _drop_hallucinations(text):
    """Drop whole sentences that are just stock ASR-hallucination filler."""
    keep = []
    for s in seg.segment(text):
        norm = _HALLUC_RE.sub("", s.lower()).strip().strip("-").strip()
        if norm and norm not in _HALLUC_SENTS:
            keep.append(s)
    return " ".join(keep).strip()


def _new_state():
    return {"buf": [], "voiced": False, "last_voice": 0.0, "transcript": "", "paras": []}


def _resample(y, sr):
    """Linear-resample mono float audio to 16 kHz (browsers usually send 48 kHz)."""
    if sr == SR or len(y) == 0:
        return y
    n = max(1, round(len(y) * SR / sr))
    xp = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
    x = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x, xp, y).astype(np.float32)


def _transcribe(audio):
    try:
        st = RECOGNIZER.create_stream()
        st.accept_waveform(SR, audio)
        RECOGNIZER.decode_stream(st)
        return st.result.text.strip()
    except Exception as exc:  # noqa: BLE001
        print("transcribe failed:", exc)
        return ""


def _chunk(transcript):
    sents = seg.segment(transcript)
    if len(sents) > 1:
        return chunk_adaptive(sents)
    return [transcript] if transcript else []


def _render(state, status=""):
    body = "".join(f"**¶ {i}** — {p}\n\n" for i, p in enumerate(state["paras"], 1))
    md = "#### 📑 paragraphs\n\n" + (body or "_start talking…_\n\n")
    return md + (f"`{status}`" if status else "")


def _flush(state):
    """Transcribe the buffered utterance, append it, and re-chunk."""
    if not state["buf"]:
        return
    audio = np.concatenate(state["buf"]).astype(np.float32)
    state["buf"], state["voiced"] = [], False
    text = _drop_hallucinations(_transcribe(audio))
    if text:
        state["transcript"] = f"{state['transcript']} {text}".strip()
        state["paras"] = _chunk(state["transcript"])


def on_mic(new_chunk, state):
    if state is None:
        state = _new_state()
    if RECOGNIZER is None:
        return "Speech model is still loading (first run downloads ~270 MB) or unavailable — try the text tabs.", state
    if new_chunk is None:                           # stream paused/ended -> flush trailing
        if state["voiced"] and sum(len(c) for c in state["buf"]) >= MIN_UTT:
            _flush(state)
        return _render(state), state
    sr, y = new_chunk
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) and np.max(np.abs(y)) > 1.5:          # int16 -> float [-1, 1]
        y = y / 32768.0
    y = _resample(y, sr)
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    now = time.time()
    if rms >= SILENCE_RMS:
        state["voiced"], state["last_voice"] = True, now
    state["buf"].append(y)
    total = sum(len(c) for c in state["buf"])
    if not state["voiced"] and total > SR * 2:      # drop leading silence
        state["buf"] = state["buf"][-2:]
        total = sum(len(c) for c in state["buf"])
    quiet = now - (state["last_voice"] or now)
    if state["voiced"] and ((quiet > PAUSE_SEC and total >= MIN_UTT) or total >= MAX_UTT):
        _flush(state)
        return _render(state), state
    return _render(state, f"🎤 listening… {total / SR:.1f}s buffered"), state


def _clear_mic():
    return "Reset — start talking…", _new_state()


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
            if (drift or len(cur) >= MAX_PARA_SENTS) and len(cur) >= MIN_SIZE:
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
        long_run = (i - start) >= MAX_PARA_SENTS          # readability cap
        if (drift or long_run) and (i - start) >= MIN_SIZE:
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
        "# 🎙️ Quint — speech → semantic paragraphs\n"
        "Speak (or paste text) and watch it transcribe and group into semantic "
        "paragraphs **on a CPU**. "
        "[Code](https://github.com/poloniki/quint) · `pip install quintessentia`"
    )
    with gr.Tab("🎙️ Live mic"):
        gr.Markdown(
            "Click the mic, allow access, and talk. It buffers your audio, and when "
            "you **pause** it transcribes the sentence (Moonshine — a fast CPU model, "
            "clean punctuated text) and folds it into the right semantic paragraph. "
            "_English; first use downloads ~270 MB, so the very first response can "
            "take a minute._"
        )
        mic = gr.Audio(sources=["microphone"], streaming=True, label="microphone")
        clear = gr.Button("Clear / restart")
        mic_out = gr.Markdown()
        mic_state = gr.State(None)
        mic.stream(
            on_mic, [mic, mic_state], [mic_out, mic_state],
            time_limit=600, stream_every=0.5, concurrency_limit=1,
        )
        clear.click(_clear_mic, None, [mic_out, mic_state])
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
