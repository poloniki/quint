import requests
import pandas as pd
import random
import seaborn as sns


def color_df(df):
    df.index += 1
    # Add styling
    cmGreen = sns.light_palette("green", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)
    df = df.style.background_gradient(
        cmap=cmGreen,
        subset=[
            "Relevancy",
        ],
    )
    format_dictionary = {
        "Relevancy": "{:.1%}",
    }
    df = df.format(format_dictionary)
    return df

def bert_df(keywords,video_id):


    df = pd.DataFrame(keywords)
    generated = [2**each/random.randint(6,20) for each in range(len(df))]
    generated.sort(reverse=True)
    df['Keys'] = df[0] + ', '+df[1]+ ', ' +df[2]+ ', '+df[3]
    df['Relevancy'] = generated
    df = df[['Keys', 'Relevancy']]
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
