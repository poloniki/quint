---
title: Quint Demo
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Quint demo (Hugging Face Space)

A live demo of [Quint](https://github.com/poloniki/quint)'s text pipeline: paste
a transcript and it splits it into semantically coherent sections and summarizes
each one. (Transcription with Whisper needs a GPU, so this CPU Space focuses on
the chunk + summarize stage.)

## Deploy this Space

1. Create a new Space at <https://huggingface.co/new-space> → SDK: **Gradio**.
2. Upload the three files from this directory (`app.py`, `requirements.txt`,
   `README.md`) to the Space repo root.
3. In the Space **Settings → Variables and secrets**, add a secret
   `OPENAI_API_KEY` (used for summaries; without it the demo still chunks).
4. The Space builds and goes live at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.

The demo installs `quintessentia` from PyPI, so it always tracks the published
package.
