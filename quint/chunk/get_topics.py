from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from quint.chunk.preprocess import clean_text
from quint.chunk.load_data import get_text_data

# Train BERTopic with a custom CountVectorizer

def get_topics(path_to_text):
    # Load text from result of prediction
    print("Start to get topics")
    text = get_text_data(path_to_text)
    # Clean sentences

    sentences_clean = clean_text(text)
    # Initiate vectorizer model
    vectorizer_model = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
    # Model topics
    topic_model = BERTopic(vectorizer_model=vectorizer_model, language="english", calculate_probabilities=True, verbose=True, nr_topics='auto', n_gram_range=(1, 3),diversity=0.5)
    #topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True, nr_topics='auto', n_gram_range=(1, 3),diversity=0.5)

    # Get main topics
    topics, probs = topic_model.fit_transform(sentences_clean)
    topics = topic_model.get_topics()


    return topics
