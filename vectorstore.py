import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings):
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_embedding, k):
        _, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )
        return indices[0]
