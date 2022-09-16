# Load neccesary libraries
from scipy.signal import butter,filtfilt
from sklearn.cluster import KMeans
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Set settings for lowpass filter
T = 50.0        # Sample Period
fs = 10.0       # sample rate, Hz
cutoff = 3      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 11 * fs   # Nyquist Frequency
order = 6       # 6  # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples


def butter_lowpass_filter(data, cutoff:int, order:int):
    """ Function returns smoothed out array of sentences cosine similarities.

    Args:
        data (list/array): array of sentences cosine similarities
        cutoff (int): desired cutoff frequency of the filter
        order (int): parameter for filter

    Returns:
        numpy array: smoothed out sentence similarities
    """

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Get the smoothed array
    y = filtfilt(b, a, data)
    return y

def get_middle_points(df:pd.DataFrame,embeddings:np.array) -> list:
    """ Function returns split points to create paragraphs in text

    Args:
        df (pandas Dataframe): dataframe should consist of sentenses and timestamps
        embeddings (numpy array): embedded sentences

    Returns:
        list: list of split points
    """

    # Creating sparse matrix for cosine similarities to be calculated
    A_sparse = sparse.csr_matrix(embeddings)
    # Get the matrix of cosine similarities of each sentence to each sentence
    similarities = cosine_similarity(A_sparse)
    # Get mean similarity of each sentence for the whole text
    mean_similarity = similarities.mean(axis=0)
    # Add column of mean similarity to the input df which contains sentences
    df['mean_similarity'] = mean_similarity

    try:
        # Smooth out cosine similarity to get rid of the noise
        df['smooth'] = butter_lowpass_filter(mean_similarity,cutoff, fs, order)
        # Create a threshold to select outliers
        outlier = df['smooth'].mean() - df['smooth'].std()
        # Filter out everything whic does not
        df['masked'] = df['smooth'] > outlier
        df['masked'] = df['masked'].astype(int)
        df = df.reset_index()

        time_diff = df.loc[df.masked==0].copy()
        time_diff['diff_cum'] = time_diff['cum'].diff().fillna(0.0)
        time_diff['outliers'] = time_diff['diff_cum'].apply(lambda x: x > time_diff['diff_cum'].mean()+time_diff['diff_cum'].std(ddof=0))
        total_categories = time_diff['outliers'].sum() +1

        # Create dataframe for selecting zones which represent lower bound of the sin-like graph
        X = df[['cum','masked']]
        kmeans = KMeans(n_clusters=total_categories, random_state=0).fit(X)
        df['label'] = kmeans.labels_
        change_points = df.groupby('label').median()['index'] -1
        true_middle_points = [min(df['index'], key=lambda x:abs(x-each)) for each in change_points]
    except:
        true_middle_points=[0]

    return true_middle_points
