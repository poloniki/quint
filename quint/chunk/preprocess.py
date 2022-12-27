import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean(text: str) -> list:
    """
    Clean and pre-process a given text string by removing punctuation, lowercasing, tokenizing,
    removing numbers, removing stop words, and lemmatizing the words.

    Parameters:
    text (str): The text to be cleaned and pre-processed.

    Returns:
    lemmatized (list): The list of lemmatized words in the text.
    """
    # Remove Punctuation
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    # Lower Case
    lowercased = text.lower()
    # Tokenize
    tokenized = word_tokenize(lowercased)
    # Remove numbers
    words_only = [word for word in tokenized if word.isalpha()]
    # Make stopword list
    stop_words = set(stopwords.words('english'))
    # Remove Stop Words
    without_stopwords = [word for word in words_only if not word in stop_words]
    # Initiate Lemmatizer
    lemma = WordNetLemmatizer()
    # Lemmatize
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords]
    return lemmatized




def clean_text(sentences: list) -> list:
    """
    Clean and pre-process a list of text strings by applying the clean() function to each string.

    Parameters:
    sentences (list): List of text strings to be cleaned and pre-processed.

    Returns:
    clean_sentences (list): List of cleaned and pre-processed text strings.
    """
    # Apply clean() function to each string in the list and join the resulting list of lemmatized words into a single string
    clean_sentences = [' '.join(clean(each)) for each in sentences]
    print('Sentences cleaned')
    return clean_sentences
