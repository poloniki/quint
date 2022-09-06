from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from quint.chunk.preprocess import clean_text


# Train BERTopic with a custom CountVectorizer

def get_topics(text):
    # Load text from result of prediction
    print("Start to get topics")
    #text = get_text_data(path_to_text)
    # Clean sentences
    text = text.replace("?", ".")
    sentences_clean = clean_text(text)
    # Initiate vectorizer model
    #vectorizer_model = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
    # Model topics
    topic_model = BERTopic(language="english", nr_topics='auto', n_gram_range=(1, 3))
    #topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True, nr_topics='auto', n_gram_range=(1, 3),diversity=0.5)

    # Get main topics
    topics = topic_model.fit_transform(sentences_clean)
    topics = topic_model.get_topics()
    print('Finish topics')
    return topics


if __name__ == '__main__':
    filename = 'data/resultsJoeShort.txt'
    text = ''
    with open (filename, 'r') as f:
        for each in f.readlines():
            text+=each
    result = get_topics(text)
    print(result)
