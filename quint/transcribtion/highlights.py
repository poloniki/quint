from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import pandas as pd
import os
import re
from quint.chunk.chunking import get_middle_points
from quint.transcribtion.words_outline import outline


SYMBOLS_STEP = os.getenv('SYMBOLS_STEP')
model = SentenceTransformer('all-MiniLM-L6-v2')

import string

def preprocessing(sentence):
    sentence = ''.join([each for each in sentence if each not in string.punctuation])
    return sentence

def create_embedding(path, version=1):
    # Read the document as a single string
    if version == 1:
        with open(path) as f:
            doc = f.read()
    else:
        doc = path.replace('/n', '\n')
    # Replace question marks with period and split into sentences
    # doc = doc.replace('?', '. ')
    # sentences = doc.split('. ')
    sentences = re.findall('.*?[.!\?]\s', doc)
    # Encode the sentences using the model
    embeddings = model.encode(sentences)
    print('Embeddings for highlights created')
    return sentences, embeddings



def create_df(sentences, embeddings):
    df = pd.DataFrame()
    df['sentence'] = sentences   #[:-1]
    df['len'] = df['sentence'].apply(lambda x: len(x))
    df['cum'] = df['len'].cumsum()
    return df

def get_best_sentences(df, embeddings):
    """
    Find the best sentences from a dataframe of sentences and their corresponding embeddings.
    The best sentences are those with the highest or lowest average cosine similarity to all other sentences in the group.
    The sentences are divided into groups based on the value of their 'cum' column in the dataframe.

    Parameters:
    df (pandas DataFrame): DataFrame containing 'sentence' and 'cum' columns.
    embeddings (numpy array): Numpy array of sentence embeddings corresponding to the sentences in the DataFrame.

    Returns:
    best_sentences (list): List of the best sentences.
    df (pandas DataFrame): DataFrame with a new 'highlight' column added, indicating whether each sentence is a best sentence.
    """
    size = np.sqrt(np.sqrt(len(df)))
    steps = round(df['cum'].sum() / round(size))
    best_sentences = []
    steps = range(0, df.cum.max() + steps, steps*2)
    for each in range(len(steps)-1):
        temp_df = df.loc[(df.cum > steps[each]) & (df.cum < steps[each+1])]
        indexes = temp_df.index
        A =  embeddings[indexes[0]:indexes[-1]+1]
        A_sparse = sparse.csr_matrix(A)
        similarities = cosine_similarity(A_sparse)
        average_sim = list(np.round(similarities.mean(axis=1), 2))
        min_score = np.min(average_sim)
        max_score = np.max(average_sim)
        best_sentence = [i for i, x in enumerate(average_sim) if (x > max_score - 0.02) | (x == min_score)]
        for every in best_sentence:
            best_sentences.append(temp_df.iloc[every].sentence)
    best_sentences = [each for each in best_sentences if len(each) > 50]
    df['highlight'] = df['sentence'].apply(lambda x: x in best_sentences)
    print('Best sentences created.')
    return best_sentences, df



def get_colored_transcript(text):
    """
    Generates a highlighted version of the input text by identifying the most important
    words and sentences in the text.

    Parameters:
    text (str): The input text.

    Returns:
    str: The highlighted version of the input text.
    """
    # Create embeddings for the input text
    sentences, embeddings = create_embedding(text, version=2)
    # Create a dataframe of sentence and embedding data
    df = create_df(sentences, embeddings)
    # Get the middle points in the dataframe
    middle_points = get_middle_points(df, embeddings)
    # Outline the most important words in the dataframe
    outline_df = outline(df)
    # Get the best sentences in the dataframe
    best_sentences, df = get_best_sentences(df, embeddings)
    # Initialize an empty string to store the highlighted text
    highlighted_text = ''
    # Iterate over the rows in the dataframe
    for i, row in df.iterrows():
        # Get the sentence and highlight flag
        sentence = row['sentence']
        highlight = row['highlight']
        # Outline the most important words in the sentence
        to_bold = outline_df['names'].iloc[i]
        if to_bold:
            to_unpack = [word.split(' ', 1) for word in to_bold]
            flat_list = [item for sublist in to_unpack for item in sublist]
            sentence = " ".join(f'<span class="BestWords">{word}</span>' if (preprocessing(word) in flat_list) and (preprocessing(word).lower() != 'the') else word for word in sentence.split())
        # Highlight the best sentences
        if highlight:
            sentence = f'<span class="BestWords">{sentence}.</span> '
        else:
            sentence = f'{sentence}. '
        # Add the sentence to the highlighted text, with chunking if necessary
        if i in middle_points:
            highlighted_text += f' \n \n {sentence}'
        else:
            highlighted_text += f'{sentence}'
    print('Colored text ready.')
    return highlighted_text
