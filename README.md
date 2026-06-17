# Quint — real-time speech → semantic paragraphs

<p align="center">
  <img src="https://raw.githubusercontent.com/poloniki/quint/master/frontend/logo.png" alt="Quint logo">
</p>

<p align="center">
  <a href="https://pypi.org/project/quintessentia/">
    <img src="https://img.shields.io/pypi/v/quintessentia?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI">
  </a>
  <a href="https://huggingface.co/spaces/poloniki/quint-demo">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Talk%20to%20it-yellow?style=for-the-badge" alt="Live demo on Hugging Face Spaces">
  </a>
  <a href="https://github.com/poloniki/quint/actions/workflows/build.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/poloniki/quint/build.yml?branch=master&style=for-the-badge&logo=github&label=CI" alt="CI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT">
  </a>
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/python-3.10-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python 3.10">
  </a>
</p>

**Quint turns speech into clean, semantically-grouped paragraphs — in real time, on a CPU, no GPU.** Talk into a microphone (or feed it a live transcript) and Quint transcribes it, restores punctuation, and groups the sentences into coherent paragraphs *the instant each one closes* — driven by a single-pass streaming chunker built on fast static embeddings. It also ships a batch API for turning podcasts, lectures, and YouTube videos into transcribed, chunked, and summarized text. Open-source and self-hostable.

## Table of Contents

- [Live demo](#-live-demo)
- [Real-time semantic chunking](#-real-time-semantic-chunking)
- [How the live demo works](#-how-the-live-demo-works)
- [Batch API](#-batch-api-transcribe--chunk--summarize)
- [Quickstart](#-quickstart)
- [License](#-license)
- [Deploy on a GPU cloud](#-deploy-the-batch-api-on-a-gpu-cloud)

## 🎬 Live demo

▶️ **[Talk to it in your browser](https://huggingface.co/spaces/poloniki/quint-demo)** — speak into your mic and watch it transcribe, punctuate, and group your words into semantic paragraphs **live, on a free CPU Space**. No install, no GPU. (There are also text tabs: paste a transcript to watch paragraphs form one pass, or chunk + summarize.)

The demo's source and auto-deploy setup live in [`huggingface-space/`](huggingface-space/).

## ⚡ Real-time semantic chunking

The heart of Quint is a **single-pass streaming chunker** ([`quint/chunking/streaming.py`](quint/chunking/streaming.py)): it reads sentences one at a time, compares each to a running topic centroid, and closes a paragraph the moment the topic drifts. No O(n²) similarity matrix and no lookahead — so it runs on a *live* stream with only bounded memory.

The split rule is **self-calibrating**: it opens a new paragraph when a sentence is anomalously dissimilar relative to the running mean/spread seen so far (`sim < mean − z·std`). The knob `z` is scale-free (standard deviations), needs no per-corpus tuning, and adapts to each document — the right behaviour for true streaming.

Pair it with [Model2Vec](https://github.com/MinishLab/model2vec) static embeddings (~30 MB, no torch, ~80 µs/sentence) for real-time chunking on a CPU:

```python
from model2vec import StaticModel
from quint.chunking.streaming import stream_chunks_live, stream_chunks_adaptive

model = StaticModel.from_pretrained("minishlab/potion-base-8M")  # fast static embeddings
def embed(sentence):
    return model.encode(sentence)

# Streaming: yields each paragraph the instant it closes (e.g. words from speech-to-text)
for paragraph in stream_chunks_live(sentence_stream, embed):
    print(paragraph)

# Batch list-in / list-out, same self-calibrating rule:
paragraphs = stream_chunks_adaptive(my_sentences, embed)
```

`stream_chunks(sentences, embed, threshold=...)` is also available if you prefer a fixed threshold. The chunker is embedding-agnostic — pass any callable mapping a sentence to a vector (it's L2-normalized internally), so `sentence-transformers`, OpenAI embeddings, etc. all work.

## 🛠 How the live demo works

The mic pipeline in [`huggingface-space/app.py`](huggingface-space/app.py) is a small, fully-local CPU stack — no GPU, no torch:

```
mic audio → buffer until a pause → Moonshine STT → restore punctuation → semantic chunker → paragraphs
```

- **Speech-to-text:** [Moonshine](https://github.com/usefulsensors/moonshine) (base.en, via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)). Its encoder scales compute with the actual audio length instead of Whisper's fixed 30 s window, so short pause-delimited utterances transcribe in tens of milliseconds.
- **Punctuation:** a CT-transformer restores sentence punctuation on Moonshine's run-ons (selectively — well-formed sentences pass through untouched), so paragraphs break at real sentence ends.
- **Chunking:** Model2Vec embeddings + the streaming chunker above.

## 🚀 Batch API (transcribe · chunk · summarize)

Quint also runs as a FastAPI service for long-form audio. Once it's running (see [Quickstart](#-quickstart)), interactive docs are at `/docs`.

| Endpoint | Does |
| --- | --- |
| `GET /youtube_transcript?video_id=…` | Fetch a YouTube video's audio and transcribe it |
| `POST /file_transcript` | Transcribe an uploaded audio file |
| `POST /chunk` `{ "body": "…" }` | Split long text into semantic paragraphs → `{ "output": ["…", "…"] }` |
| `POST /best_sentence` `{ "body": "…" }` | Index of the most descriptive sentence |
| `GET /youtube_summarize?video_id=…` | Chunked per-section summaries of a video |

## 🧑‍💻 Quickstart

Install from PyPI (the import package is `quint`):

```shell
pip install quintessentia
```

For real-time CPU chunking, also grab the fast static embeddings:

```shell
pip install model2vec
```

Or run the batch API locally from source — CPU is fine for chunking and summarization; Whisper transcription is far faster on a GPU (see [deploy](#-deploy-the-batch-api-on-a-gpu-cloud)).

```shell
git clone https://github.com/poloniki/quint.git
cd quint
make install              # pip install -e .
cp env.sample .env        # then set OPENAI_API_KEY
make run_api              # serves on http://localhost:8083
```

Then open `http://localhost:8083/docs` for the interactive API docs.

### Web UI (optional)

A small [Streamlit](https://streamlit.io) frontend lives in [`frontend/`](frontend/app.py). With the API running:

```shell
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Set `QUINT_API_URL` if the API isn't on `http://localhost:8083`.

## 📖 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🛜 Deploy the batch API on a GPU cloud

The real-time chunker and the live demo run anywhere on a CPU. Only the **batch Whisper transcription** benefits from a GPU. For that, I recommend the JAX backend — it's much faster than the OpenAI-proposed way. See [Whisper JAX](https://github.com/sanchit-gandhi/whisper-jax); one of its tables:

**Table 1:** Average inference time in seconds for audio files of increasing length. GPU device is a single A100 40GB GPU.
TPU device is a single TPU v4-8.

<div align="center">

|           | OpenAI  | Transformers | Whisper JAX | Whisper JAX |
| --------- | ------- | ------------ | ----------- | ----------- |
|           |         |              |             |             |
| Framework | PyTorch | PyTorch      | JAX         | JAX         |
| Backend   | GPU     | GPU          | GPU         | TPU         |
|           |         |              |             |             |
| 1 min     | 13.8    | 4.54         | 1.72        | 0.45        |
| 10 min    | 108.3   | 20.2         | 9.38        | 2.01        |
| 1 hour    | 1001.0  | 126.1        | 75.3        | 13.8        |
|           |         |              |             |             |

</div>

### Choosing a GPU cloud provider

Quint runs on any machine with an NVIDIA GPU, so you are free to use whichever cloud provider (AWS, GCP, Azure, Lambda, Paperspace, RunPod, …) or on-prem hardware you prefer. For the best price/performance on transcription, look for an **Ada-generation card** such as the RTX 6000 Ada or A6000 — these are typically far cheaper than A100-class GPUs while offering comparable [CUDA compute capability](https://developer.nvidia.com/cuda-gpus).

Whatever you pick, you only need an instance that provides:

- An **NVIDIA GPU** (Ampere/Ada or newer recommended)
- **Ubuntu 22.04** (or similar) with **CUDA 12** and **Docker**
- **SSH access** (root or sudo)

The steps below are provider-neutral: provision the instance however your provider requires, then follow along.

### 1. Configure your environment

```shell
cp env.sample .env        # then edit .env
direnv reload             # or: source .env
```

Set the following in `.env`:

| Variable | Used by | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | API (summarization) | Key for the summarization step |
| `GPU_TYPE` | API (optional) | Set to `A100` to enable bfloat16 on the JAX backend; any other value (or unset) uses float16 |
| `EMBEDDING_MODEL` | API (optional) | sentence-transformer used for chunking; default `all-MiniLM-L6-v2`. Use a multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) for non-English text |
| `SEGMENT_LANGUAGE` | API (optional) | pysbd sentence-split language; default `en` (e.g. `ja` for Japanese) |
| `EMAIL` | deploy helper | Labels / generates your SSH key |
| `HOST` | deploy helper | Public IP or hostname of your GPU instance |
| `SSH_USER` | deploy helper | SSH login user for your image (often `root`, but `ubuntu` on AWS, your username on GCP, `azureuser` on Azure) |

### 2. Provision and connect to the instance

Create a GPU instance with your provider using an **Ubuntu 22.04 + CUDA 12 + Docker** image and your SSH public key. Once it is running, note its public IP (set it as `HOST` in `.env`) and connect:

```shell
ssh $SSH_USER@$HOST -i ~/.ssh/<your_key>
```

> Use the login user your provider specifies for the image. `root` works on many bare-VM providers, but AWS Ubuntu AMIs use `ubuntu`, GCP uses your username, Azure uses `azureuser`, etc. Set it as `SSH_USER` in `.env`.

The notebook [`notebooks/Deploy_gpu_instance.ipynb`](notebooks/Deploy_gpu_instance.ipynb) automates the provider-neutral parts: generating an SSH key, copying the code to the host, and building/running the container.

### 3. Install NVIDIA drivers (if your image doesn't include them)

If the instance image already ships with working drivers, skip this. Otherwise run the bundled script on the instance and reboot to load them:

```shell
bash scripts/install_nvidia_driver.sh
sudo reboot
```

### 4. Get the code onto the instance

Clone it directly:

```shell
git clone https://github.com/poloniki/quint.git
cd quint
```

…or copy your local checkout up with `scp` (the deploy notebook does this for you).

### 5. Build and run

```shell
docker build -t quint --file Dockerfile.jax .
docker run --gpus all -p 80:80 --shm-size=1g --env-file .env quint
```

> The `--env-file .env` flag passes `OPENAI_API_KEY` (and optional `GPU_TYPE`) into the container, so make sure `.env` is present on the instance. Also ensure your provider's firewall / security group allows inbound TCP on port **80** — most clouds only open SSH (port 22) by default.

Your API is now available on the instance's public IP (port 80).
