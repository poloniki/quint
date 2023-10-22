from pydub import AudioSegment
from pydub.utils import which
import os
from quint.tools.time import timed
from quint.params import *

# Setting up the converter for pydub
AudioSegment.converter = which("ffmpeg")


@timed
def convert_mp4_to_flac(audio_file_name: str) -> str:
    """
    Convert a MP4 audio file to FLAC format.

    Parameters:
    - audio_file_name (str): The name of the MP4 audio file (without path).

    Returns:
    - str: Path of the converted FLAC audio file.

    Notes:
    - The MP4 file should be located in the directory specified by AUDIO_PATH.
    - The converted FLAC file will be saved in the directory specified by AUDIO_PATH.

    Raises:
    - FileNotFoundError: If the source MP4 file is not found.
    - Exception: If there's an issue during conversion.
    """

    # Constructing the full paths for source and destination
    src_path = f"{AUDIO_PATH}/{audio_file_name}"
    dst_file_name = audio_file_name.rsplit(".", 1)[0] + ".flac"
    dst_path = f"{AUDIO_PATH}/{dst_file_name}"

    # Check if the source file exists
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    try:
        sound = AudioSegment.from_file(src_path, format="mp4")
        sound.export(dst_path, format="flac")
        # Delete the input MP4 file
        os.remove(src_path)
    except Exception as e:
        raise Exception(f"Error converting {src_path} to FLAC. Error: {e}")

    return dst_path
