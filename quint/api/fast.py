from cgitb import text
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Response
from pydantic import BaseModel
import datetime
import re
from pydub import AudioSegment
from quint.transcribtion import google_api as tga
from quint.transcribtion import highlights
from quint.chunk.get_topics import get_topics
from quint.chunk.timestamp import get_timestamp
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

@app.get("/")
def root():
    return {'greeting': 'Hello'}


@app.post("/transcript")
def upload(file: UploadFile = File(...)):
    audio_file_name= file.filename
    if audio_file_name not in os.listdir("."):
        print('We got a new file')
        try:
            contents = file.file.read()
            with open(file.filename, 'wb') as f:
                # Get audio file name
                audio_file_name = audio_file_name.split('.')[0] + '.wav'
                # if audio_file_name not in os.listdir("."):
                # Save audio file locally
                f.write(contents)
                f.close()
            # # Get audio file transcribtion
            transcript = tga.google_transcribe(audio_file_name)

            if len(transcript) > 30000000:
                try:
                    topics = get_topics(transcript)
                except:
                    topics = 'Text is too short.'
            else:
                topics = 'Text is too short.'
            # Get colored highlights
            transcript = highlights.get_colored_transcript(transcript)
            # Create name for transcript
            transcript_filename = audio_file_name.split('.')[0] + '.txt'
            # Save transript file locally
            tga.write_transcripts(transcript_filename ,transcript)

            # Return transcript to the api query
            return  {'transcript' : transcript, 'topics':topics}


        except Exception as error:
            return {"message": error}

        finally:
            file.file.close()

    with open(output_filepath+file.filename.split('.')[0] + '.txt') as f:
        # Get audio file name
        transcript = f.readlines()

        f.close()

    return {'transcript':transcript}



class Body(BaseModel):
    text: str


@app.post("/chunk")
def chunking_text(body: Body):
    input_text = body.text

    #Clean version without most importan words and sentences
    sentences,embeddings = create_embedding(input_text , version=2)
    df = create_df(sentences,embeddings)
    true_middle_points=get_middle_points(df,embeddings)
    #Initiate text
    text=''
    for num, each in enumerate(df['sentence']):
        # Chunk the text
        if num in true_middle_points:
            text+=f' \n \n {each}. '
        else:
            text+=f'{each}. '
    clean_chunks = text.split('\n \n')
    return {'for_summary':clean_chunks}


# class BodyList(BaseModel):
#     transcript: list = [dict]
#     chunks:list = [str]

# from fastapi import Query
# from typing import List

# @app.post("/timestamp")
# def getting_timestamps(chunks:List[str], transcript:List[dict]):



#     text = ''
#     for i in transcript:
#         text += ' ' + f'{[i["start"]]} ' + highlights.preprocessing(i['text'])

#     final_dict = {}
#     for i, each in enumerate(chunks):
#         timestamp = get_timestamp(highlights.preprocessing(each[0:500].lower()), highlights.preprocessing(text).lower(), 3)
#         timestamp = round(float(re.findall("\d+\.\d+", timestamp)[0]))
#         timestamp = str(datetime.timedelta(seconds=timestamp))
#         final_dict.update({i:{timestamp:each}})

#     return final_dict
