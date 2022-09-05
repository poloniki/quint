# Filter requirements.
import numpy as np
from scipy.signal import butter,filtfilt
from sklearn.cluster import KMeans
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


T = 50.0        # Sample Period
fs = 10.0       # sample rate, Hz
cutoff = 3      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 11 * fs   # Nyquist Frequency
order = 6       # 6 # 10     # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples


def butter_lowpass_filter(data, cutoff, fs, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

def get_middle_points(df,embeddings):
    A_sparse = sparse.csr_matrix(embeddings)
    similarities = cosine_similarity(A_sparse)
    story = similarities.mean(axis=0)
    df['story'] = story
    story = similarities.mean(axis=0)
    try:
        df['smooth'] = butter_lowpass_filter(story,cutoff, fs, order)
        df['smooth'] = df['smooth'].apply(lambda x: round(round(x*100,2)**2,0))
        outlier = df['smooth'].mean() - df['smooth'].std()
        df['masked'] = df['smooth'] > outlier
        df['masked'] =df['masked'].astype(int)
        df = df.reset_index()

        time_diff = df.loc[df.masked==0].copy()
        time_diff['diff_cum'] = time_diff['cum'].diff().fillna(0.0)
        time_diff['outliers'] = time_diff['diff_cum'].apply(lambda x: x > time_diff['diff_cum'].mean()+time_diff['diff_cum'].std(ddof=0))
        total_categories = time_diff['outliers'].sum() +1

        X = df[['cum','masked']]
        kmeans = KMeans(n_clusters=total_categories, random_state=0).fit(X)
        df['label'] = kmeans.labels_
        change_points = df.groupby('label').median()['index'] -3
        true_middle_points = [min(df['index'], key=lambda x:abs(x-each)) for each in change_points]
    except:
        true_middle_points=[0]

    return true_middle_points
