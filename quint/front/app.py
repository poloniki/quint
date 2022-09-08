from distutils import text_file
import streamlit as st
import base64
import time
import random
from youtube_transcript_api import YouTubeTranscriptApi
from IPython.display import YouTubeVideo
import os
#Importing from our own repository
from processing import concatenate_lines
from punctuation_api import punctuate
from chunk_api import chunk
from summary_api import summarize
from timestamp import timestamping
from youtube import video_name
from getting_best_api import get_best
from bert import get_bert, color_df
import pandas as pd
import random


def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)



layout="wide"
st.set_page_config(page_title='Quint')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.session_state['flag'] = False



# component= "<h1 style= 'color: red' > Inject Header HTML  </h1>"
# st.markdown(component, unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.15,0.65,0.1])

with col1:
    st.write(' ')

with col2:
    st.image("logo.png")

with col3:
    st.write(' ')



@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

        # background-color: green;
        # background-image: linear-gradient(to right, #99ff99, #99ff99);




main_html = '''
    <style>
    thead tr th:first-child {display:none}
    tbody th {display:none}

    .css-a51556 {
        border-bottom: 1px solid rgba(49, 51, 63, 0.1);
        border-right: 1px solid rgba(49, 51, 63, 0.1);
        vertical-align: middle;
        padding: 0.25rem 0.375rem;
        font-weight: 600;
        color: rgb(12 12 12 / 80%);
        background-color: rgb(12 12 12 / 4%);
    }

    .css-1n76uvr {
        width: 900px;
        position: relative;
        display: flex;
        flex: 1 1 0%;
        flex-direction: column;
        gap: 1rem;
    }

    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;

    }

    .st-cs {
        background-color: green;
        background-image: linear-gradient(to right, #dd3764, #99ff99);
    }

    .TimeStamp {
        background: rgb(200 230 254 / 80%);
        padding: 0.12em 0.3em;
        margin: 0 0.0000001em;
        line-height: 1;
        border-radius: 0.15em;
        text-decoration: underline;
    }
    .BestWords {
        background: #feca74;
        padding: 0.12em 0.3em;
        margin: 0 0.0000001em;
        line-height: 1;
        border-radius: 0.15em"
    }

    .css-1cpxqw2 {
        color:green;
        width:8em;
        height:2.5em;
        padding-bottom: 0 rem;
        padding-top: 100 rem;
        align: buttom;

    .ccss-ocqkz7 e1tzin5v4 {

        padding-bottom: 0 rem;
        padding-top: 0 rem;
        align: buttom;


    </style>
    '''
st.markdown(main_html, unsafe_allow_html=True)


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



if 'status' not in st.session_state:
    st.session_state['status'] = 'submitted'


def refresh_state():
	st.session_state['status'] = 'submitted'


col1, col2 = st.columns([0.8,0.2])


with col1:
    link = st.text_input('', 'https://www.youtube.com/watch?v=7ZUSVabB6N4', on_change=refresh_state)


with col2:
    st.text('')
    st.text('')
    summary = st.button('QUINT')


if 'seconds' not in st.session_state:
	st.session_state.seconds = 0

#https://www.youtube.com/watch?v=6T7pUEZfgdI&t=6586s
# button = """<a href="https://www.youtube.com/watch?v=6T7pUEZfgdI&t=6586s" class="button">Go to Google</a>"""
# st.markdown(button, unsafe_allow_html=True)


video_id = link.split("=")[1]
YouTubeVideo(video_id)

text_file = f"{video_id}.txt"

_col1, _col2 = st.columns([0.7,0.3])

# First we check if we already have summary for some specifict podcast
if (summary) & (text_file not in os.listdir("results/")):
    # VISUAL ELEMENT - starting
    progress = 0
    my_bar = st.progress(progress)

    ### GETTING TRANSCRIPT ####
    # Get the transcipt: list of dictionaries
    YouTubeTranscriptApi.get_transcript(video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    my_bar.progress(progress + 1)

    # Now we need to get the text out and punctuate it
    concatenated_text = concatenate_lines(transcript)  ## Function that creates txt file from list of texts
    my_bar.progress(progress + 4)
    print('Starting punctuating')
    punctuated_text = punctuate(concatenated_text) ## Punctuate concatenated text
    my_bar.progress(progress + 30)
    with open(f'transcripts/{video_id}_transcript.txt', 'w') as f:
        f.write(punctuated_text)
        f.close()
    # Now we need to chunk text into main parts
    chunked_text = chunk(punctuated_text) ## Returns list of chunks
    print(f'Chunked into {len(chunked_text)} chunks.')
    my_bar.progress(progress + 15)

    # Get the timestamp of each chunk
    timestamps = timestamping(chunked_text,transcript)
    my_bar.progress(progress + 5)
    ###### SUMMARIZATION PART #########
    # Create a list of ready summaries
    summary_list = [summarize(each) for each in chunked_text]
    #Take out escape characters and punct which messes with API
    summary_list = list(map(lambda chunk : chunk.replace('\\','').replace('\"','') , summary_list))
    my_bar.progress(progress + 30)



    # Create headlines from the summaries
    headlines =[summarize(each, length=50) for each in summary_list]
    headlines = list(map(lambda chunk : chunk.replace('\\','').replace('\"','') , headlines))
    my_bar.progress(progress + 10)

    # Add headlines to the summaries + get best with higlights
    summary_list =[f"<b>{headlines[i]}</b>" + '\n\n' + get_best(each) for i,each in enumerate(summary_list)]
    my_bar.progress(progress + 5)


    # # Create Html tags for best words and sents
    # summary_list = [ for each in summary_list]
    #Add timestamps to summaries
    summary_list = [f"<span class='TimeStamp'><a href='https://www.youtube.com/watch?v={video_id}?t={get_sec(timestamps[i])}'>{timestamps[i]}</a></span>" + ' ' + each for i,each in enumerate(summary_list)]



    # Create an concatenated summary from summary_list

    # Join the resulting summary list into one text
    title = video_name(video_id)
    summary = '\n\n'.join(summary_list)

    summary = f"<h2>{title}</h2> \n\n\n {summary}"

    #Get bert topics as DF
    topics = get_bert(punctuated_text,video_id)



    # Return the result to streamlit


    with _col1:
        st.markdown(summary, unsafe_allow_html=True)
    with _col2:
        st.table(topics)

    #### SAVE TEXT FILE TO PROCESS LATER ####
    with open(f'results/{video_id}.txt', 'w') as f:
        f.write(summary)
        f.close()


    # option = st.selectbox(
    #     'Select timstamp',
    #     (each for each in timestamps),on_change=)

    #st.write('You selected:', option)


    st.video(link)

# If we already have the summary of some video - return it
elif summary:

    # Imitate loading progress
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(random.uniform(0, 0.1))
        my_bar.progress(percent_complete + 1)

    summary = ''
    with open(f"results/{text_file}", 'r') as f:
        for line in f.readlines():
            summary+=line
    topics = pd.read_csv(f'topics/{video_id}.csv',index_col=0)
    topics = color_df(topics)


    with _col1:
        st.markdown(summary, unsafe_allow_html=True)
    with _col2:
        st.table(topics)
    st.video(link)
