from quint.tools.activations import rev_sigmoid
import numpy as np

# Library to import pre-trained model for sentence embeddings
# Calculate similarities between sentences
from sklearn.metrics.pairwise import cosine_similarity

# package for finding local minimas
from scipy.signal import argrelextrema


def activate_similarities(similarities: np.array, p_size=10) -> np.array:
    """Function returns list of weighted sums of activated sentence similarities
    Args:
        similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
        p_size (int): number of sentences are used to calculate weighted sum
    Returns:
        list: list of weighted sums
    """
    # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
    x = np.linspace(-10, 10, p_size)
    # Then we need to apply activation function to the created space
    y = np.vectorize(rev_sigmoid)
    # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
    activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))
    ### 1. Take each diagonal to the right of the main diagonal
    diagonals = [
        similarities.diagonal(each) for each in range(0, similarities.shape[0])
    ]
    ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
    diagonals = [
        np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals
    ]
    ### 3. Stack those diagonals into new matrix
    diagonals = np.stack(diagonals)
    ### 4. Apply activation weights to each row. Multiply similarities with our activation.
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    ### 5. Calculate the weighted sum of activated similarities
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities


def get_middle_points(embeddings: np.array) -> list:
    # Create similarities matrix
    similarities = cosine_similarity(embeddings)

    # Let's apply our function. For long sentences i reccomend to use 10 or more sentences
    activated_similarities = activate_similarities(similarities, p_size=10)

    ### 6. Find relative minima of our vector. For all local minimas and save them to variable with argrelextrema function
    minmimas = argrelextrema(
        activated_similarities, np.less, order=2
    )  # order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
    return minmimas
