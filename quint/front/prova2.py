from turtle import right
import streamlit as st
import base64
import youtube_dl
import requests
import pprint
from time import sleep
from pytube import YouTube
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from IPython.display import YouTubeVideo

st.session_state['flag'] = False


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }

    .stProgress .st-bo {
        background-color: green;
    }

    .stButton{
        color:maroon;
        width:3em;
        height:3em;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('sfondo_prova.jpg')

def main_page():
    st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

    st.markdown("# Quintessentia 🎈")
    st.sidebar.markdown("# Quintessentia🎈")

def page2():
    #st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# About us 🎉")
    st.sidebar.markdown("# About us 🎉")

page_names_to_funcs = {
    "Quintessentia": main_page,
    "Classify NOW!": page2,
    "About us": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()



if 'status' not in st.session_state:
    st.session_state['status'] = 'submitted'

ydl_opts = {
   'format': 'bestaudio/best',
   'postprocessors': [{
       'key': 'FFmpegExtractAudio',
       'preferredcodec': 'mp3',
       'preferredquality': '192',
   }],
   'ffmpeg-location': './',
   'outtmpl': "./%(id)s.%(ext)s",
}
CHUNK_SIZE = 5242880


@st.cache
def transcribe_from_link(link, categories: bool):
	_id = link.strip()

	def get_vid(_id):
		with youtube_dl.YoutubeDL(ydl_opts) as ydl:
			return ydl.extract_info(_id)

	# download the audio of the YouTube video locally
	meta = get_vid(_id)
	save_location = meta['id'] + ".mp3"

	print('Saved mp3 to', save_location)






def read_file(filename):
	with open(filename, 'rb') as _file:
		while True:
			data = _file.read(CHUNK_SIZE)
			if not data:
				break
			yield data


	# upload audio file to AssemblyAI
	upload_response = requests.post(
		upload_endpoint,
		headers=headers_auth_only, data=read_file(save_location)
	)

	audio_url = upload_response.json()['upload_url']
	print('Uploaded to', audio_url)

	# start the transcription of the audio file
	transcript_request = {
		'audio_url': audio_url,
		'iab_categories': 'True' if categories else 'False',
	}

	transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)

	# this is the id of the file that is being transcribed in the AssemblyAI servers
	# we will use this id to access the completed transcription
	transcript_id = transcript_response.json()['id']
	polling_endpoint = transcript_endpoint + "/" + transcript_id

	print("Transcribing at", polling_endpoint)

	return polling_endpoint


def get_status(polling_endpoint):
	polling_response = requests.get(polling_endpoint, headers=headers)
	st.session_state['status'] = polling_response.json()['status']

def refresh_state():
	st.session_state['status'] = 'submitted'



link = st.text_input('Enter your YouTube video link', 'https://www.youtube.com/watch?v=efs3QRr8LWw', on_change=refresh_state)
_left, mid, _right = st.columns(3)

with _right:
   st.video(link)

st.text("The transcription is " + st.session_state['status'])


video_id = link.split("=")[1]
YouTubeVideo(video_id)

with _left:
   if st.button('Transcript NOW!', disabled=False):
        YouTubeTranscriptApi.get_transcript(video_id)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        result = ""
        for i in transcript:
            result += ' ' + i['text']
        #print(result)
        print(len(result))
        text_result= result

        num_iters = int(len(result)/1000)

        for i in range(0, num_iters + 1):
            start = 0
            start = i * 1000
            end = (i + 1) * 1000
            print("input text \n" + result[start:end])

        st.balloons()
        st.markdown(result)

with mid:
    my_bar = st.progress(0)
    summary = st.button('Summary')
    if summary:
        YouTubeTranscriptApi.get_transcript(video_id)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        st.session_state['flag'] = not st.session_state['flag']


        print(st.session_state['flag'])
        if st.session_state['flag']:
            print('I am sending the request')
            result = ""
            for i in transcript:
                result += ' ' + i['text']
            url =  'https://bart-finetuned-summarization-6u4yq4wz5q-no.a.run.app/generate'
            myobj = {'text': result, 'summary_min_length': 0, 'summary_max_length': 1000}
            with requests.Session() as session:
                with session.post(url, json = myobj , stream=True) as response:
                    for idx , line in enumerate(response.iter_lines()):
                        my_bar.progress((idx+10)*10)
                        print(line)
                        st.markdown(line)


"""Created by Francesco Puglia"""
