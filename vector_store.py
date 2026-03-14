# vector_store.py

import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.metadata = []

    def add(self, embeddings, texts, metadatas):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)
        self.metadata.extend(metadatas)

    def search(self, query_embedding, top_k=3):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"),
            top_k
        )
        
        results = []
        for idx in indices[0]:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx]
            })
        return results