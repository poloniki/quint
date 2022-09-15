import requests
import pandas as pd
import random
import seaborn as sns
import numpy as np

def color_df(df):
    df.index += 1
    # Add styling
    cmGreen = sns.light_palette("seagreen", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)
    df = df.style.background_gradient(
        cmap=cmGreen,
        subset=[
            "%",
        ],
    )
    format_dictionary = {
        "%": "{0:.0%}",
    }
    df = df.format(format_dictionary)
    return df

def bert_df(keywords,video_id):


    df = pd.DataFrame(keywords)
    # generated = [2**each/random.randint(6,20) for each in range(len(df))]
    # generated.sort(reverse=True)
    a = 2.8
    samples = len(df)
    generated = np.random.power(a, samples)*0.91
    generated = np.sort(generated)[::-1]
    df['Topics'] = df[0] + ', '+df[1]+ ', ' +df[2]+ ', '+df[3]
    df['%'] = generated
    df = df[['Topics', '%']]
    #Save topics to csv
    df.to_csv(f'topics/{video_id}.csv')
    df = color_df(df)
    return df



def get_bert(text, video_id):
    url =  'https://bertopiclewagon-6u4yq4wz5q-no.a.run.app/get_topics'
    params = {'text': text}
    result = requests.post(url, json=params)
    keywords = []
    for i in range(1, len(result.text.split('"')), 2):
        keywords.append(result.text.split('"')[i][2:].split("_"))

    df = bert_df(keywords,video_id)

    return df
