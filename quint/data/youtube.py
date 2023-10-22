from pytube import YouTube
from quint.params import *


def download_youtube_video(video_id: str) -> str:
    """
    Download the audio stream of a given YouTube video by its video ID.

    Parameters:
    - video_id (str): The ID of the YouTube video.

    Returns:
    - str: The name of the downloaded audio file in .wav format.

    Raises:
    - Exception: If there's an issue downloading the video or if no audio streams are found.
    """

    youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        youtube = YouTube(youtube_url)
    except Exception as e:
        raise Exception(f"Error fetching video from URL: {youtube_url}. Error: {e}")

    audio_streams = youtube.streams.filter(only_audio=True)

    if not audio_streams:
        raise Exception(f"No audio streams found for video: {youtube_url}")

    # Download the audio stream and get the file path
    audio_file_path = audio_streams[0].download(AUDIO_PATH)

    # Extract the file name without its extension
    file_name_without_ext = audio_file_path.split("/")[-1].rsplit(".", 1)[0]

    # Append .wav extension
    audio_file_name = f"{file_name_without_ext}.mp4"

    return audio_file_name
