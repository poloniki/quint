---
title: Quint — real-time speech to paragraphs
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "6.18.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Quint — real-time speech → semantic paragraphs

A live demo of [Quint](https://github.com/poloniki/quint), running entirely on a
**free CPU Space, no GPU**. Speak into your mic and it transcribes (Moonshine via
sherpa-onnx), restores punctuation, and groups your words into semantic paragraphs
on the fly with Model2Vec static embeddings + Quint's single-pass streaming chunker.
Two more tabs work without a mic: paste a transcript to watch paragraphs form one
pass, or chunk + summarize a transcript.

## Deploy

### Automatic (recommended)

This folder auto-deploys via
[`.github/workflows/sync-space.yml`](../.github/workflows/sync-space.yml): on
every push it creates the Space (if needed) and uploads these files. One-time
setup:

1. Create a Hugging Face token with **write** access:
   <https://huggingface.co/settings/tokens>.
2. In the GitHub repo → **Settings → Secrets and variables → Actions**, add a
   secret named `HF_TOKEN`. The `HF_SPACE_ID` variable is preset to
   `poloniki/quint-demo` (change it there for a different name).
3. Push a change in this folder, or run the *Sync demo to Hugging Face Space*
   workflow manually. The Space is created and goes live.
4. On the Space → **Settings → secrets**, add `OPENAI_API_KEY` to enable
   summaries (without it the demo still chunks).

### Manual

Create a Gradio Space at <https://huggingface.co/new-space> and upload `app.py`,
`requirements.txt`, and this `README.md` to its root.

The demo is self-contained: it mirrors Quint's streaming chunker with only the
lightweight, torch-free deps it needs (model2vec, sherpa-onnx, pysbd, numpy,
openai), so it builds and runs on a free CPU Space. See the full package at
[`quintessentia`](https://pypi.org/project/quintessentia/).
