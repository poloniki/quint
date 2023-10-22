import pysbd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
seg = pysbd.Segmenter(language="en", clean=True)


def create_embedding(text: str = None):
    # Split sentencesgi
    sentences = seg.segment(text)
    # Encode the sentences using the model
    embeddings = model.encode(sentences)
    print("Embeddings for highlights created")
    return sentences, embeddings
