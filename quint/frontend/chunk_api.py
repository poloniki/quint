import requests

def chunk(text):
    url = "https://imagequint-cro4ll255q-ew.a.run.app/chunk"
    params = {'text':text}
    response = requests.post(url, json=params)
    chunks = response.json()['for_summary']
    return chunks
