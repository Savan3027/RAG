# retrieval.py

def retrieve(query, embedder, store, top_k=3):
    query_embedding = embedder(query)
    return store.search(query_embedding, top_k)