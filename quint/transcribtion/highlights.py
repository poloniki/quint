from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import pandas as pd
from termcolor import colored
import os

SYMBOLS_STEP = os.getenv('SYMBOLS_STEP')
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embedding(path , version = 1):
    if version == 1:
        with open(path) as f:
            doc = f.readlines()
    else:
        doc = path.split('/n')
    doc = ' '.join(doc)

    doc = doc.replace("?", ". ")
    sentences = doc.split('. ')
    embeddings = model.encode(sentences)
    print('Embeddings for highlights created')
    return sentences, embeddings


def create_df(sentences, embeddings):
    df = pd.DataFrame()
    df['sentence'] = sentences[:-1]
    df['len'] = df['sentence'].apply(lambda x: len(x))
    df['cum'] = df['len'].cumsum()
    return df

def get_best_sentences(df,embeddings):


    best_sentences = []
    steps = range(0, df.cum.max() + 8000, 8000)
    for each in range(len(steps)-1):
        temp_df = df.loc[(df.cum > steps[each])&(df.cum < steps[each+1])]
        indexes = temp_df.index
        A =  embeddings[indexes[0]:indexes[-1]+1]
        A_sparse = sparse.csr_matrix(A)
        similarities = cosine_similarity(A_sparse)
        average_sim = list(np.round(similarities.mean(axis=1),2))
        min_score =np.min(average_sim)
        max_score = np.max(average_sim)
        best_sentence = [i for i, x in enumerate(average_sim) if (x > max_score - 0.02) | (x == min_score)]
        for every in best_sentence:
            best_sentences.append(temp_df.iloc[every].sentence)

    best_sentences = [each for each in best_sentences if len(each) > 50]
    df['highlight'] = df['sentence'].apply(lambda x: x in best_sentences)
    print('Best sentences created.')

    return best_sentences, df


def get_colored_transcript(text):
    sentences, embeddings = create_embedding(text , version = 2)
    df = create_df(sentences, embeddings)
    best_sentences, df = get_best_sentences(df,embeddings)
    text = ''
    for num, each in enumerate(df['sentence']):
        if df['highlight'].iloc[num] == True:
            text+=colored(f'{each}. ','white','on_red')
        else:
            text+=f'{each}. '
    print('Colored text ready.')
    return text
