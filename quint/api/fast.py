import os
import shutil
import logging

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from quint.data.youtube import download_youtube_video
from quint.preprocessing.audio import convert_mp4_to_flac
from quint.transcribtion.transcription import transcribe_flac
from quint.params import *
from quint.transcribtion.transcriber import WhisperTranscriber
from quint.chunking.generate import get_chunks

logging.basicConfig(level=logging.INFO)

app = FastAPI()


app.state.transcriber = WhisperTranscriber()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class Body(BaseModel):
    body: str = None


# Main page welcome greeting
@app.get("/")
def root():
    """Welcome endpoint.

    Returns:
        dict: dictionary with welcome string as a value
    """
    return {"greeting": "Welcome to Podcast Summarization API!"}


@app.post("/transcript")
def upload(file: UploadFile = File(...)):
    """
    Endpoint which allows you to upload an audio file and get its transcript in text format.

    Parameters:
    file (UploadFile, optional): Audio file of WAV or FLAC format.

    Returns:
    dict: Dictionary with transcript string as a value.
    """
    # Get the audio filename
    audio_file_name = file.filename
    # Get the transcriber object
    transcriber = app.state.transcriber

    # Update audio file name to have an .mp4 extension and determine the full path
    audio_file_name = audio_file_name.split(".")[0] + ".mp4"
    src_path = f"{AUDIO_PATH}/{audio_file_name}"

    with open(src_path, "wb") as f:
        # Stream the file contents directly to disk using shutil
        shutil.copyfileobj(file.file, f)

    converted_flac_path = convert_mp4_to_flac(audio_file_name)
    transcription_results = transcribe_flac(
        transcriber=transcriber,
        flac_path=converted_flac_path,
    )
    return {"transcript": transcription_results}


@app.post("/chunk")
def chunking_text(body: Body):
    """
    This endpoint takes a block of text and splits it into reasonable chunks.

    Parameters:
    body (Body): A dictionary containing the text to be chunked.

    Returns:
    dict: Returns a list of text paragraphs.
    """
    # Extract text from the endpoint input
    input_text = body.body
    chunks = get_chunks(input_text)

    return {"output": chunks}


@app.post("/best")
def highlight_words(body: Body):
    """This endpoint takes text ad input and returns text with highlighted: best sentences (most descriptive ones), names, products, companies,
    dates.

    Args:
        body: row text

    Returns:
        dict: returns a dictionary where value is a text with highlights
    """
    # Extract text from the endpoint input
    input_text = body.body
    # Get the highlighted transcript
    transcript = highlights.get_colored_transcript(input_text)
    return {"edited": transcript}


@app.get("/youtube_transcript")
def upload(
    video_id: str = "WdTeDXsOSj4",
):
    transcriber = app.state.transcriber
    print(type(transcriber))

    audio_file_name = download_youtube_video(video_id=video_id)
    converted_flac_path = convert_mp4_to_flac(audio_file_name)
    transcription_results = transcribe_flac(
        transcriber=transcriber,
        flac_path=converted_flac_path,
    )
    return {"transcript": transcription_results}
