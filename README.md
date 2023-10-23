# Quint

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
