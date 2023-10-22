# from bertopic import BERTopic
# from sklearn.feature_extraction.text import TfidfVectorizer
# from quint.chunk.preprocess import clean_text

# def get_topics(text: str) -> list:
#     """
#     Extract main topics from a block of text.

#     Parameters:
#     text (str): A block of text from which to extract topics.

#     Returns:
#     list: A list of main topics extracted from the text.
#     """
#     print("Start to get topics")
#     # Clean sentences
#     sentences_clean = clean_text(text)
#     # Initiate the BERTopic model
#     topic_model = BERTopic(language="english", nr_topics='auto', n_gram_range=(1, 3))
#     # Fit the model to the text and get the main topics
#     topics = topic_model.fit_transform(sentences_clean)
#     topics = topic_model.get_topics()
#     print('Finish topics')
#     return topics


# if __name__ == '__main__':
#     filename = 'data/resultsJoeShort.txt'
#     text = ''
#     with open (filename, 'r') as f:
#         for each in f.readlines():
#             text+=each
#     result = get_topics(text)
#     print(result)
