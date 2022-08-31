from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from quint.transcribtion import google_api as tga
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


    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            # Save audio file locally
            f.write(contents)
            # Get audio file namec
            audio_file_name= file.filename
            audio_file_name = audio_file_name.split('.')[0] + '.wav'
            # Get audio file transcribtion
            transcript = tga.google_transcribe(audio_file_name)
            # Create name for transcript
            transcript_filename = audio_file_name.split('.')[0] + '.txt'
            # Save transript file locally
            tga.write_transcripts(transcript_filename,transcript)

            # Return transcript to the api query
            return transcript


    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}
