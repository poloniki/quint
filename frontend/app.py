"""Streamlit web UI for the Quint API.

Run it from the repo root with the API already serving:

    pip install -r frontend/requirements.txt
    streamlit run frontend/app.py

Point it at a different backend with the QUINT_API_URL environment variable
(defaults to http://localhost:8083).
"""

import os
import re

import requests
import streamlit as st

API_URL = os.environ.get("QUINT_API_URL", "http://localhost:8083")
LOGO = os.path.join(os.path.dirname(__file__), "logo.png")

# Transcription downloads + runs Whisper, so give those calls a generous timeout.
TRANSCRIBE_TIMEOUT = 600
QUICK_TIMEOUT = 60

st.set_page_config(page_title="Quint", page_icon="🎙️", layout="centered")


def extract_video_id(value: str) -> str | None:
    """Pull an 11-character YouTube video ID out of a URL or a raw ID."""
    value = value.strip()
    if not value:
        return None
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", value)
    return match.group(1) if match else None


def api_request(method: str, path: str, base_url: str, *, timeout: int, **kwargs):
    """Call the API and return (ok, payload_or_error_message)."""
    url = f"{base_url.rstrip('/')}{path}"
    try:
        resp = requests.request(method, url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return True, resp.json()
    except requests.exceptions.ConnectionError:
        return False, f"Could not reach the API at {base_url}. Is it running?"
    except requests.exceptions.Timeout:
        return False, "The request timed out — transcription can take a few minutes."
    except requests.exceptions.HTTPError as exc:
        return False, f"API error: {exc.response.status_code} {exc.response.reason}"
    except ValueError:
        return False, "The API returned a non-JSON response."


with st.sidebar:
    st.header("Settings")
    base_url = st.text_input("API URL", value=API_URL)
    if st.button("Check connection"):
        ok, payload = api_request("GET", "/", base_url, timeout=QUICK_TIMEOUT)
        st.success("Connected ✅") if ok else st.error(payload)

if os.path.exists(LOGO):
    st.image(LOGO, use_container_width=True)
st.markdown(
    "<p style='text-align:center'>Transcribe · chunk · summarize podcasts</p>",
    unsafe_allow_html=True,
)

tab_summary, tab_transcript, tab_chunk = st.tabs(
    ["📝 Summarize", "🎥 Transcript", "📜 Chunk text"]
)

with tab_summary:
    st.subheader("Summarize a YouTube video")
    url = st.text_input(
        "YouTube URL or video ID",
        value="https://www.youtube.com/watch?v=WdTeDXsOSj4",
        key="summary_url",
    )
    if st.button("Summarize", type="primary"):
        video_id = extract_video_id(url)
        if not video_id:
            st.warning("Please enter a valid YouTube URL or 11-character video ID.")
        else:
            with st.spinner("Downloading, transcribing and summarizing…"):
                ok, payload = api_request(
                    "GET",
                    "/youtube_summarize",
                    base_url,
                    params={"video_id": video_id},
                    timeout=TRANSCRIBE_TIMEOUT,
                )
            if ok:
                summaries = payload.get("summary", [])
                st.success(f"{len(summaries)} summary chunk(s)")
                for i, chunk in enumerate(summaries, start=1):
                    st.markdown(f"**Part {i}**")
                    st.write(chunk)
            else:
                st.error(payload)

with tab_transcript:
    st.subheader("Transcribe a YouTube video")
    url = st.text_input(
        "YouTube URL or video ID",
        value="https://www.youtube.com/watch?v=WdTeDXsOSj4",
        key="transcript_url",
    )
    if st.button("Transcribe"):
        video_id = extract_video_id(url)
        if not video_id:
            st.warning("Please enter a valid YouTube URL or 11-character video ID.")
        else:
            with st.spinner("Downloading and transcribing…"):
                ok, payload = api_request(
                    "GET",
                    "/youtube_transcript",
                    base_url,
                    params={"video_id": video_id},
                    timeout=TRANSCRIBE_TIMEOUT,
                )
            if ok:
                transcript = payload.get("transcript")
                if isinstance(transcript, dict):
                    transcript = transcript.get("text", transcript)
                st.text_area("Transcript", value=str(transcript), height=400)
            else:
                st.error(payload)

with tab_chunk:
    st.subheader("Split text into semantic chunks")
    text = st.text_area("Paste text to chunk", height=200, key="chunk_text")
    if st.button("Chunk"):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Chunking…"):
                ok, payload = api_request(
                    "POST", "/chunk", base_url, json={"body": text}, timeout=QUICK_TIMEOUT
                )
            if ok:
                chunks = payload.get("output", [])
                st.success(f"{len(chunks)} chunk(s)")
                for i, chunk in enumerate(chunks, start=1):
                    with st.expander(f"Chunk {i}"):
                        st.write(chunk)
            else:
                st.error(payload)
