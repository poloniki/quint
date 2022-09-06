import requests

def summarize(text):
    url =  'https://bart-finetuned-summarization-6u4yq4wz5q-no.a.run.app/generate'
    params = {'text': text, 'summary_min_length': 0, 'summary_max_length': 300}

    summary = requests.post(url, json=params)

    return summary.text
