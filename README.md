# Quint

"Quint" is designed to enhance the podcast experience. It simplifies the process for users, making it easier for them to understand and navigate podcasts by providing concise summaries, highlights, and transcripts.

## 🚀 Main Functionality

Below is a list of the core API endpoints offered by Quint:

### 🎥 YouTube Video Transcription

Endpoint: /youtube_transcript?video_id=YOUR_YOUTUBE_VIDEO_ID
Simply provide a YouTube video ID. Quint will fetch the video, extract its audio content, and return a transcription of the audio.

`GET /youtube_transcript?video_id=YOUR_YOUTUBE_VIDEO_ID`

### 🎙️ Transcription from Audio File

** Endpoint: /file_transcript **
Upload an audio file and instantly receive its transcription in text format.

`POST /file_transcript`

### 📜 Text Chunking

** Endpoint: /chunk **
Submit a lengthy text and get it divided into semantically meaningful chunks or paragraphs.

`POST /chunk`

### 🌟 Highlight the Best Sentences

** Endpoint: /best_sentence **
Submit a text and let Quint analyze it. The endpoint returns the index of the most descriptive sentence based on the embeddings.

`POST /best_sentence`

### 📖 License

This project is licensed under the MIT License - see the LICENSE.md file for details.

Feel free to adjust as necessary, and replace placeholders like the documentation link as appropriate.
