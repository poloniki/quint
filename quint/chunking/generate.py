from quint.chunking.similarities import get_middle_points
from quint.tools.embedding import create_embedding


def get_chunks(input_text):
    """
    Split the input text into reasonable chunks based on semantic similarity.

    Parameters:
    - input_text (str): The text to be chunked.

    Returns:
    - list of str: Chunks of text formed from the input text.

    Raises:
    - ValueError: If the input text is empty.
    - RuntimeError: If there is an error in generating embeddings or finding split points.
    """

    if not input_text:
        raise ValueError("The provided input text is empty.")

    try:
        sentences, embeddings = create_embedding(input_text, is_path=False)
    except Exception as e:
        raise RuntimeError(f"Error generating embeddings for the input text. {str(e)}")

    try:
        split_points = get_middle_points(embeddings)[0]
    except Exception as e:
        raise RuntimeError(f"Error finding split points for chunking. {str(e)}")

    # Generate text chunks based on provided split points
    chunks = []
    chunk = ""

    for idx, sentence in enumerate(sentences):
        chunk += sentence
        if idx in split_points:
            chunks.append(chunk.strip())
            chunk = ""

    if chunk:
        chunks.append(chunk.strip())

    return chunks
