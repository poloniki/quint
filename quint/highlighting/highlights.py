from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_best_sentence_index(embeddings: np.array) -> int:
    """
    Find the index of the best sentence based on embeddings.
    The best sentence is the one with the highest average cosine similarity to all other sentences.

    Parameters:
    embeddings (numpy array): Numpy array of sentence embeddings.

    Returns:
    best_sentence_index (int): Index of the best sentence.
    """
    # Calculate cosine similarities for embeddings
    similarities = cosine_similarity(embeddings)

    # Calculate the average similarity of each sentence to all others
    average_sim = similarities.mean(axis=1)

    # Return the index of the sentence with the highest average similarity
    best_sentence_index = int(np.argmax(average_sim))

    return best_sentence_index
