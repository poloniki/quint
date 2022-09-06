from google.cloud import speech_v1 as speech
from pydub import AudioSegment
import io
import os
from google.cloud import speech
import wave
from google.cloud import storage
import time
import threading
#import soundfile
import wave



filepath = os.getenv('AUDIO_INPUT_PATH')
bucketname = os.getenv('BUCKETNAME')
output_filepath = os.getenv('OUTPUP_PATH')

def mp3_to_wav(audio_file_name):
    if audio_file_name.split('.')[1] == 'mp3':
        sound = AudioSegment.from_mp3(audio_file_name)
        audio_file_name = audio_file_name.split('.')[0] + '.wav'
        sound.export(audio_file_name, format="wav")
        print(u'\u2713', ' Converted to wav')


def stereo_to_mono(audio_file_name):
    time.sleep(0.5)
    print('Start Mono')
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")
    print('Converted to mono')


def frame_rate_channel(audio_file_name):
    print('Start channels')
    stereo_to_mono(audio_file_name)
    with wave.open(audio_file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        print('Got channels', frame_rate,channels)

        return frame_rate,channels

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print('Uploaded_file')


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()


def google_transcribe(audio_file_name):
    file_name = filepath + audio_file_name
    mp3_to_wav(file_name)

    # The name of the audio file to transcribe
    frame_rate, channels = frame_rate_channel(file_name)

    if channels > 1:
        stereo_to_mono(file_name)

    bucket_name = bucketname
    source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name

    upload_blob(bucket_name, source_file_name, destination_blob_name)

    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name

    print('Started transcript')
    transcript = ''

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config   = speech.RecognitionConfig(
    encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=frame_rate,
    language_code='en-US',
    enable_automatic_punctuation=True,
    model='latest_long')

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)

    response = operation.result(timeout=10000)
    print('Got the response')

    for result in response.results:
        transcript += result.alternatives[0].transcript

    delete_blob(bucket_name, destination_blob_name)

    return transcript

def write_transcripts(transcript_filename,transcript):
    f=open(output_filepath + transcript_filename,"w+")
    f.write(transcript)
    f.close()

if __name__ == '__main__':
    audio_file_name= 'JoeShort.wav'
    transcript = google_transcribe(audio_file_name)
    transcript_filename = audio_file_name.split('.')[0] + '.txt'
    write_transcripts(transcript_filename,transcript)

# if __name__ == "__main__":
#     for audio_file_name in os.listdir(filepath):
