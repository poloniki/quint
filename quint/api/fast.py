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
from quint.highlighting.highlights import get_best_sentence_index
from quint.tools.embedding import create_embedding

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
    return {"greeting": "Welcome to the Quint API!"}


@app.post("/file_transcript")
def upload(file: UploadFile = File(...)):
    """
    Endpoint to upload an audio file and receive its transcription in text format.

    Parameters:
    file (UploadFile, optional): Audio file to be transcribed. Although the endpoint supports various audio formats like WAV and FLAC,
                                 it converts and processes them as MP4 for internal use.

    Returns:
    dict: Returns a dictionary containing the transcription of the audio content from the uploaded file.
          Example: {"transcript": "The transcribed text of the audio goes here..."}

    Usage:
        POST /file_transcript with the audio file as part of the request body.

    Note:
        Ensure that you have the necessary permissions and rights to transcribe the audio content of any file you provide.
    """

    # Get the audio filename and the transcriber object
    audio_file_name = file.filename
    transcriber = app.state.transcriber

    # Update the audio filename to have an .mp4 extension and compute the full path to store the file
    audio_file_name = audio_file_name.split(".")[0] + ".mp4"
    src_path = f"{AUDIO_PATH}/{audio_file_name}"

    with open(src_path, "wb") as f:
        # Stream the file contents directly to the specified path
        shutil.copyfileobj(file.file, f)

    # Convert the saved audio to FLAC format
    converted_flac_path = convert_mp4_to_flac(audio_file_name)

    # Get the transcription of the audio content
    transcription_results = transcribe_flac(
        transcriber=transcriber,
        flac_path=converted_flac_path,
    )

    return {"transcript": transcription_results}


@app.post("/chunk")
def chunking_text(body: Body):
    """
    Endpoint that takes a continuous block of text and divides it into semantically meaningful chunks or paragraphs.

    Parameters:
    body (Body): Request body containing the raw text to be chunked.
                 The body should have a key named 'body' holding the text.

    Returns:
    dict: A dictionary containing the list of chunked paragraphs derived from the input text.
          Example: {"output": ["Chunk 1", "Chunk 2", ...]}

    Usage:
        POST /chunk
        {
            "body": "Your lengthy continuous text here..."
        }

    Note:
        The function uses semantic analysis to divide the text, ensuring that each chunk or paragraph is contextually coherent.
    """
    # Extract text from the endpoint input
    input_text = body.body

    # Split the input text into coherent chunks
    chunks = get_chunks(input_text)

    return {"output": chunks}


@app.post("/best_sentence")
def best_sentence(body: Body):
    """
    Endpoint that analyzes the given text to identify and highlight the most descriptive sentences.
    Currently, it returns the best sentence index based on the embeddings.

    Args:
        body (Body): Request body containing the raw text to be analyzed.
                     The body should have a key named 'body' holding the text.

    Returns:
        dict: A dictionary containing the index of the most descriptive sentence in the input text.
              Example: {"best_sentence_index": 5}

    Usage:
        POST /best_sentence
        {
            "body": "Your raw text here..."
        }

    Note:
        The function currently identifies the best sentence based on embeddings. Future enhancements
        could add more highlighted entities like names, products, companies, and dates.
    """
    # Extract text from the endpoint input
    input_text = body.body

    # Get the highlighted transcript
    _, embedded_sentences = create_embedding(input_text)
    best_sentence = get_best_sentence_index(embedded_sentences)

    return {"best_sentence_index": best_sentence}


@app.get("/youtube_transcript")
def youtube_transcript(video_id: str = "WdTeDXsOSj4"):
    """
    Endpoint that fetches a YouTube video based on its video ID, extracts its audio content, and returns a transcription of the audio.

    Parameters:
    video_id (str, optional): The unique video ID associated with a YouTube video. Default is set to "WdTeDXsOSj4".

    Returns:
    dict: A dictionary containing the transcription of the audio from the specified YouTube video.
          Example: {"transcript": "The transcribed text of the video goes here..."}

    Usage:
        GET /youtube_transcript?video_id=YOUR_YOUTUBE_VIDEO_ID

    Note:
        Ensure that you have the necessary permissions and rights to transcribe the audio content of any video you provide.
    """

    transcriber = app.state.transcriber

    # Download audio from the provided YouTube video ID
    audio_file_name = download_youtube_video(video_id=video_id)

    # Convert the downloaded audio to FLAC format
    converted_flac_path = convert_mp4_to_flac(audio_file_name)

    # Get the transcription of the audio content
    transcription_results = transcribe_flac(
        transcriber=transcriber,
        flac_path=converted_flac_path,
    )

    return {"transcript": transcription_results}
