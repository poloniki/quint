import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embedding(path: str = None, is_path: bool = True):
    # Read the document as a single string
    if is_path == True:
        with open(path) as f:
            doc = f.read()
    else:
        doc = path.replace("/n", "\n")
    # Replace question marks with period and split into sentences
    # doc = doc.replace('?', '. ')
    # sentences = doc.split('. ')
    sentences = re.findall(".*?[.!\?]\s", doc)
    # Encode the sentences using the model
    embeddings = model.encode(sentences)
    print("Embeddings for highlights created")
    return sentences, embeddings
