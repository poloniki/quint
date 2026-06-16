# Quint: transcribe | chunk | summarize

<p align="center">
  <img src="frontend/logo.png" alt="Logo">
</p>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  </a>
</p>

"Quint" is designed to enhance the podcast experience. It simplifies the process for users, making it easier for them to understand and navigate podcasts by providing concise summaries, highlights, and transcripts.

## 🚀 Main Functionality

Below is a list of the core API endpoints offered by Quint:

### 🎥 YouTube Video Transcription

Simply provide a YouTube video ID. Quint will fetch the video, extract its audio content, and return a transcription of the audio.

`GET /youtube_transcript?video_id=YOUR_YOUTUBE_VIDEO_ID`

### 🎙️ Transcription from Audio File

Upload an audio file and instantly receive its transcription in text format.

`POST /file_transcript`

### 📜 Text Chunking

Submit a lengthy text and get it divided into semantically meaningful chunks or paragraphs.

`POST /chunk`

### 🌟 Highlight the Best Sentences

Submit a text and let Quint analyze it. The endpoint returns the index of the most descriptive sentence based on the embeddings.

`POST /best_sentence`

### 📖 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## 🛜 How to deploy this API on cloud

Importand note: I highly reccomend using the JAX solution, as it is much faster than OpenAI proposed way. Please refer to this git [Whisper Jax](https://github.com/sanchit-gandhi/whisper-jax) for more details. I will attach one of the table from this git repo:

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
