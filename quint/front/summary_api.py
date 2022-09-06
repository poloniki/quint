import requests
from chunk_api import chunk

def summarize(text, length=300):
    sub_chunks = []
    if len(text.split()) > 900:
        sub_chunks = chunk(text)
        counter = 0
        for i in sub_chunks:
            sub_chunks[counter] = get_summary(i)
            counter = counter + 1
        sub_chunks_joined = ' '.join(sub_chunks)
        summary = get_summary(sub_chunks_joined)
        return summary
    summary = get_summary(text,length=300)
    return summary

def get_summary(text, length=300):
    url =  'https://bart-finetuned-summarization-6u4yq4wz5q-no.a.run.app/generate'
    params = {'text': text, 'summary_min_length': 0, 'summary_max_length': length}

    summary = requests.post(url, json=params)

    return summary.text
