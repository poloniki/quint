import pysbd
from sentence_transformers import SentenceTransformer
from typing import Tuple, List

model = SentenceTransformer("all-MiniLM-L6-v2")
seg = pysbd.Segmenter(language="en", clean=False)


def create_embedding(text: str = None) -> Tuple[List[str], List[List[float]]]:
    """
    Create embeddings for a given text.

    This function splits the provided text into sentences and then generates embeddings for each sentence using a pre-trained model.

    Parameters:
    - text (str, optional): The input text to generate embeddings for. Defaults to None.

    Returns:
    - Tuple[List[str], List[List[float]]]: A tuple where the first element is a list of sentences and the second element is a list of embeddings for each sentence.

    Example:
    >>> sentences, embeddings = create_embedding("Hello world. My name is Nikita.")
    """
    # Split sentences
    sentences = seg.segment(text)
    # Encode the sentences using the model
    embeddings = model.encode(sentences)
    return sentences, embeddings
