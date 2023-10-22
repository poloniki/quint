import os
from quint.tools.time import timed
from quint.transcribtion.transcriber import WhisperTranscriber


@timed
def transcribe_flac(flac_path: str, transcriber: WhisperTranscriber) -> list:
    """
    Transcribes the given FLAC audio using the provided WhisperTranscriber instance.

    Parameters:
    - flac_path (str): Path to the FLAC file to be transcribed.
    - transcriber (WhisperTranscriber): An instance of WhisperTranscriber to transcribe the audio.

    Returns:
    - list: A list of transcription results directly from the Whisper pipeline.

    Raises:
    - Exception: If an error occurs during transcription or if there's an issue with deleting the FLAC file.
    """
    try:
        transcription_results = transcriber.transcribe(flac_path)

        # Delete the input FLAC file
        os.remove(flac_path)

        return transcription_results

    except Exception as e:
        raise Exception(f"Error during FLAC transcription or file deletion. Error: {e}")
