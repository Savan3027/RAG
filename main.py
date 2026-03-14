from openai import OpenAI

client = OpenAI()

def generate_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


# main.py

from ingestion import extract_text_from_pdf, clean_text
from chunking import chunk_text
from embedding import get_embedding
from vector_store import FAISSStore
from retrieval import retrieve
from generation import build_prompt, generate_answer
from logger import log_query

file_path = "data/Resume.pdf"

text = extract_text_from_pdf(file_path)
text = clean_text(text)
chunks = chunk_text(text)

embeddings = []
texts = []
metadatas = []

for chunk in chunks:
    emb = get_embedding(chunk["text"])
    embeddings.append(emb)
    texts.append(chunk["text"])
    metadatas.append({
        "document": "notes.pdf",
        "chunk_id": chunk["chunk_id"]
    })

dim = len(embeddings[0])
store = FAISSStore(dim)
store.add(embeddings, texts, metadatas)

query = input("Ask a question: ")

retrieved = retrieve(query, get_embedding, store)
prompt = build_prompt(query, retrieved)
answer = generate_answer(prompt)

log_query(query, retrieved, answer)

print(answer)