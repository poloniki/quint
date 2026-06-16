"""Gradio demo of Quint's text pipeline, for Hugging Face Spaces.

Splits a transcript into semantically coherent sections and (when an
OPENAI_API_KEY Space secret is set) summarizes each one. Transcription (Whisper)
needs a GPU, so this CPU-friendly demo focuses on the chunk + summarize stage —
see https://github.com/poloniki/quint for the full self-hosted API.
"""

import os

import gradio as gr

from quint.chunking.generate import get_chunks
from quint.summarizing.summarizer import TextSummarizer

HAS_KEY = bool(os.environ.get("OPENAI_API_KEY"))
summarizer = TextSummarizer()

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
        chunks = get_chunks(text)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline error to the user
        return f"Chunking failed: {exc}"

    header = f"**{len(chunks)} section(s)**"
    if not HAS_KEY:
        header += "  \n_Set an `OPENAI_API_KEY` Space secret to enable summaries._"

    parts = [header, ""]
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"### Section {i}")
        if HAS_KEY:
            try:
                parts.append(f"**Summary:** {summarizer.summarize(chunk)}")
            except Exception as exc:  # noqa: BLE001
                parts.append(f"_(summary unavailable: {exc})_")
        parts.append(f"> {chunk}")
        parts.append("")
    return "\n".join(parts)


demo = gr.Interface(
    fn=process,
    inputs=gr.Textbox(lines=12, label="Transcript / long text", value=SAMPLE),
    outputs=gr.Markdown(label="Chunked sections + summaries"),
    title="Quint — chunk & summarize",
    description=(
        "Paste a transcript; Quint splits it into semantic sections and summarizes "
        "each. Transcription (Whisper) needs a GPU — see the repo for the full API."
    ),
    article="Powered by [Quint](https://github.com/poloniki/quint) · `pip install quintessentia`",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
