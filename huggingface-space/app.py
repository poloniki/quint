"""Gradio demo of Quint's text pipeline, for Hugging Face Spaces.

Splits a transcript into semantically coherent sections (the same algorithm as
Quint's `get_chunks`) and, when an OPENAI_API_KEY Space secret is set, summarizes
each one. It's self-contained — only the lightweight chunking/summarization deps,
not the full package — so it builds on a free CPU Space. Transcription (Whisper)
needs a GPU; see https://github.com/poloniki/quint for the full self-hosted API.
"""

import os

import gradio as gr
import numpy as np
import pysbd
from scipy.signal import argrelextrema
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

HAS_KEY = bool(os.environ.get("OPENAI_API_KEY"))

# Loaded once at startup (small, CPU-friendly).
_model = SentenceTransformer("all-MiniLM-L6-v2")
_seg = pysbd.Segmenter(language="en", clean=False)


def _rev_sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(0.5 * x))


def _activate_similarities(similarities: np.ndarray, p_size: int = 10) -> np.ndarray:
    """Quint's activation: weight each sentence by its neighbours' similarity."""
    p_size = min(p_size, similarities.shape[0])
    x = np.linspace(-10, 10, p_size)
    activation_weights = np.pad(
        np.vectorize(_rev_sigmoid)(x), (0, similarities.shape[0] - p_size)
    )
    diagonals = [similarities.diagonal(each) for each in range(similarities.shape[0])]
    diagonals = [
        np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals
    ]
    diagonals = np.stack(diagonals) * activation_weights.reshape(-1, 1)
    return np.sum(diagonals, axis=0)


def chunk_text(text: str) -> list[str]:
    """Split text into semantic chunks at local minima of activated similarity."""
    sentences = _seg.segment(text)
    if len(sentences) < 3:
        return [text.strip()]
    embeddings = _model.encode(sentences)
    activated = _activate_similarities(cosine_similarity(embeddings), p_size=10)
    minima = set(argrelextrema(activated, np.less, order=2)[0].tolist())

    chunks, current = [], ""
    for idx, sentence in enumerate(sentences):
        current += sentence
        if idx in minima:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks


def summarize(text: str) -> str:
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Summarize this in one sentence:\n\n{text}"}
        ],
    )
    return resp.choices[0].message.content.strip()


SAMPLE = (
    "Welcome back to the show. Today we're talking about how small habits compound "
    "over time. A lot of people assume big results come from big, dramatic changes. "
    "In reality, most lasting change comes from tiny actions repeated daily. Think "
    "about reading ten pages a day. On its own that feels trivial. But over a year "
    "that's more than a dozen books. The same logic applies to saving money or "
    "learning a language. The hard part isn't the size of the habit, it's the "
    "consistency. So how do you stay consistent? The first trick is to make the "
    "habit obvious. Put the book on your pillow, lay out your running shoes the "
    "night before. The second trick is to make it small enough that you can't say "
    "no. Two minutes is the magic number for most people. Once you start, momentum "
    "usually carries you further. Another big factor is your environment. We like "
    "to think we rely on willpower, but environment quietly shapes most of our "
    "choices. If junk food is on the counter, you'll eat it. If it's in a cupboard "
    "you never open, you won't. Designing your surroundings is often easier than "
    "fighting your own impulses. Finally, track your streak. Seeing a chain of good "
    "days makes you reluctant to break it. That visual feedback is surprisingly "
    "powerful. So start tiny, stay consistent, and let the compounding do the work."
)


def process(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Please paste some text first."
    try:
        chunks = chunk_text(text)
    except Exception as exc:
        return f"Chunking failed: {exc}"

    header = f"**{len(chunks)} section(s)**"
    if not HAS_KEY:
        header += "  \n_Set an `OPENAI_API_KEY` Space secret to enable summaries._"

    parts = [header, ""]
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"### Section {i}")
        if HAS_KEY:
            try:
                parts.append(f"**Summary:** {summarize(chunk)}")
            except Exception as exc:
                parts.append(f"_(summary unavailable: {exc})_")
        parts.append(f"> {chunk}")
        parts.append("")
    return "\n".join(parts)


demo = gr.Interface(
    fn=process,
    inputs=gr.Textbox(lines=12, label="Transcript / long text", value=SAMPLE),
    outputs=gr.Markdown(),
    title="Quint — chunk & summarize",
    description=(
        "Paste a transcript; Quint splits it into semantic sections and summarizes "
        "each. Transcription (Whisper) needs a GPU — see the "
        "[repo](https://github.com/poloniki/quint) for the full API. "
        "`pip install quintessentia`"
    ),
)

if __name__ == "__main__":
    demo.launch()
