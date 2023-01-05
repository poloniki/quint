import os
import shutil
import logging

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from quint.chunk.chunking import get_middle_points
from quint.transcribtion import google_api as tga
from quint.transcribtion import highlights
from quint.transcribtion.highlights import create_embedding, create_df

output_filepath = os.getenv('OUTPUT_PATH')

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

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
    """ Welcome endpoint.

    Returns:
        dict: dictionary with welcome string as a value
    """
    return {'greeting': 'Welcome to Podcast Summarization API!'}


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
    # Check if we already have this file locally
    if audio_file_name not in os.listdir("."):
        try:
            # Save the audio file locally
            with open(file.filename, 'wb') as f:
                # Get audio file name and change its format to wav
                audio_file_name = audio_file_name.split('.')[0] + '.wav'
                # Stream the file contents directly to disk using shutil
                shutil.copyfileobj(file.file, f)
            # Get audio file transcription
            transcript = tga.google_transcribe(audio_file_name)
            # Highlight: best sentences best sentences (most descriptive ones), names, products, companies, dates
            transcript = highlights.get_colored_transcript(transcript)
            # Create name for transcript
            transcript_filename = audio_file_name.split('.')[0] + '.txt'
            # Save transcript file locally
            tga.write_transcripts(transcript_filename, transcript)
            return {'transcript': transcript}
        # Catch an error and display it to the user
        except Exception as error:
            return {"message": error}
    # If we already had this audio file, then return the ready transcript to the user
    transcript = []
    # Read the transcript file line by line
    with open(output_filepath+file.filename.split('.')[0] + '.txt') as f:
        for line in f:
            transcript.append(line)
    return {'transcript': transcript}


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
    # Split text to sentences and get their embedding, we use version=2 here because it specifies that the input is text
    sentences, embeddings = create_embedding(input_text, version=2)
    # Create a dataframe which returns sentences with generated timestamps
    df = create_df(sentences, embeddings)
    # Get the points where we need to split the text
    true_middle_points = get_middle_points(embeddings)
    # Initiate text to append to
    text = ''
    for num, each in enumerate(df['sentence']):
        # Chunk the text
        # If the index of the row is equal to a split point, add two new lines to the text
        if num in true_middle_points[0]:
            text += f' \n \n {each} '
        else:
            # Append a new line of text with no new line if it is not in the splitting points list
            text += f'{each} '
    # Split text by new lines notation to get a list of texts - paragraphs
    clean_chunks = text.split('\n \n')
    return {'output': clean_chunks}


@app.post("/best")
def highlight_words(body: Body):
    """ This endpoint takes text ad input and returns text with highlighted: best sentences (most descriptive ones), names, products, companies,
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
    return {'edited': transcript}
