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

#Importing from our own repository
from processing import concatenate_lines
from punctuation_api import punctuate
from chunk_api import chunk
from summary_api import summarize


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

set_background('background.jpeg')

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


def refresh_state():
	st.session_state['status'] = 'submitted'



link = st.text_input('Enter your YouTube video link', 'https://www.youtube.com/watch?v=efs3QRr8LWw', on_change=refresh_state)
_left, _right = st.columns(2)

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

my_bar = st.progress(0)
summary = st.button('Summary')
if summary:

    # Get the transcipt: list of dictionaries
    YouTubeTranscriptApi.get_transcript(video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # Now we need to get the text out and punctuate it
    concatenated_text = concatenate_lines(transcript)  ## Function that creates txt file from list of texts
    print('Starting punctuating')
    punctuated_text = punctuate(concatenated_text) ## Punctuate concatenated text

    # Now we need to chunk text into main parts
    chunked_text = chunk(punctuated_text) ## Returns list of chunks
    print('Chunked')
    # Create a list of ready summaries
    summary_list = [summarize(each) for each in chunked_text]
    print(len(summary_list))
    # Create an concatenated summary from summary_list
    summary = '\n\n'.join(summary_list)


    # Return the result to streamlit
    st.markdown(summary)




"""Created by Francesco Puglia"""
