import requests


def get_best(text):
    url = "https://imagequint-cro4ll255q-ew.a.run.app/best"
    params = {"text": text}
    response = requests.post(url, json=params)
    best = response.json()["edited"]
    return best


import torch

torch.version.cuda
