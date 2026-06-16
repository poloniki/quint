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

The demo installs `quintessentia` from PyPI, so it always tracks the published
package.
