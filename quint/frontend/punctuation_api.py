import requests

def punctuate(text):
    url = 'https://punctuation2-6u4yq4wz5q-no.a.run.app/generate'
    params = {'text':text}
    x = requests.post(url, json=params)
    return x.text
