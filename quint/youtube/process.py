from youtube_transcript_api import YouTubeTranscriptApi



def get_youtube_transcript(url:str):
    video_id = url.split("=")[1]
    YouTubeTranscriptApi.get_transcript(video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript
