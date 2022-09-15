from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from quint.transcribtion import google_api as tga
from quint.transcribtion import highlights
from quint.chunk.chunking import get_middle_points
from quint.transcribtion.highlights import create_embedding,create_df
import os
output_filepath = os.getenv('OUTPUP_PATH')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
    """ Endpoint which allows you to upload audio and get transcript in a text format.

    Args:
        file (UploadFile, optional): Audio file of WAV or FLAC format.

    Returns:
        dict: dictionary with transcript string as a value
    """
    # Get the audio filename
    audio_file_name= file.filename
    # Check if we already have this file locally
    if audio_file_name not in os.listdir("."):
        try:
            # Read the file first
            contents = file.file.read()
            with open(file.filename, 'wb') as f:
                # Get audio file name and change its format to wav
                audio_file_name = audio_file_name.split('.')[0] + '.wav'
                # Save audio file locally
                f.write(contents)
                f.close()


            # Get audio file transcribtion
            transcript = tga.google_transcribe(audio_file_name)

            # Highlight: best sentences best sentences (most descriptive ones), names, products, companies, dates
            transcript = highlights.get_colored_transcript(transcript)
            # Create name for transcript
            transcript_filename = audio_file_name.split('.')[0] + '.txt'
            # Save transript file locally
            tga.write_transcripts(transcript_filename ,transcript)

            return  {'transcript' : transcript}

        # Catch an error and display it to user
        except Exception as error:
            return {"message": error}

        finally:
            file.file.close()

    # If we already had this audio file than return ready transcript to the user
    with open(output_filepath+file.filename.split('.')[0] + '.txt') as f:
        # Get audio file name
        transcript = f.readlines()
        f.close()

    return {'transcript':transcript}

# Create class that can take input as a text in a post request
class Body(BaseModel):
    text: str


@app.post("/chunk")
def chunking_text(body: Body):
    """ This endpoint takes row text and splits it into reasonable chunks.

    Args:
        str: takes text in the form of string

    Returns:
        dict: returns list of text paragraphs
    """
    # Extract text from the endpoint input
    input_text = body.text
    # Split text to sentences and get their embedding, we use version=2 here because it specifies, that input is a text
    sentences,embeddings = create_embedding(input_text , version=2)
    # Create dataframe which returns sentences with generated timestamps
    df = create_df(sentences,embeddings)

    # Get the points where we need to split the text
    true_middle_points=get_middle_points(df,embeddings)
    #Initiate text to append to
    text=''
    for num, each in enumerate(df['sentence']):
        # Chunk the text
        # If index of row is equal to split point than add two new lines to the text
        if num in true_middle_points:
            text+=f' \n \n {each}. '
        else:
        # Append new line of text with no new line if it is not the splitting points list
            text+=f'{each}. '

    # Split text by new lines notation to get list of texts - paragraphs
    clean_chunks = text.split('\n \n')
    return {'for_summary':clean_chunks}


@app.post("/best")
def highligh_words(body: Body):
    """ This endpoint takes text ad input and returns text with highlighted: best sentences (most descriptive ones), names, products, companies,
    dates.

    Args:
        str: row text

    Returns:
        dict: returns a dictionary where value is a text with highlights
    """
    # Extract text from the endpoint input
    input_text = body.text
    # Get the highlighted transcript
    transcript = highlights.get_colored_transcript(input_text)
    return {'edited':transcript}
