import streamlit as st
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
import faiss
from dotenv import load_dotenv
import os

st.set_page_config(
    page_title="✨ AI PDF Assistant",
    page_icon="✨",
    layout="centered"
)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

st.write("API loaded:", api_key[:10])   # temporary debug line

if not api_key:
    st.error("OpenAI API key not found in .env file")
    st.stop()

client = OpenAI(api_key=api_key)

st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(270deg, #ff6ec4, #7873f5, #4facfe, #43e97b);
    background-size: 800% 800%;
    animation: gradientShift 20s ease infinite;
}

@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.glass-card {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    margin-top: 20px;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #f0f0f0;
    margin-bottom: 30px;
}

.chat-bubble-user {
    background: #ffffff;
    padding: 15px 20px;
    border-radius: 20px;
    margin-bottom: 15px;
    color: black;
}

.chat-bubble-bot {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    padding: 20px;
    border-radius: 20px;
    color: white;
    margin-top: 10px;
}

.citation {
    font-size: 14px;
    opacity: 0.8;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">✨ AI PDF Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a PDF and chat with your document</div>', unsafe_allow_html=True)

if "index" not in st.session_state:
    st.session_state.index = None
if "texts" not in st.session_state:
    st.session_state.texts = []
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "answer" not in st.session_state:
    st.session_state.answer = None
if "question" not in st.session_state:
    st.session_state.question = None
if "citations" not in st.session_state:
    st.session_state.citations = []

def extract_text(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({
            "text": text[start:end],
            "chunk_id": chunk_id
        })
        start += chunk_size - overlap
        chunk_id += 1
    return chunks

def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)

    except Exception as e:
        st.error(f"Embedding error: {e}")
        st.stop()

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing document..."):
        text = extract_text(uploaded_file).strip()

        st.write("Extracted text length:", len(text))

        if len(text) == 0:
            st.error("PDF contains no readable text.")
            st.stop()

        chunks = chunk_text(text)

        embeddings = []
        texts = []
        metadata = []

        for chunk in chunks:
            if chunk["text"].strip() == "":
                continue

            emb = get_embedding(chunk["text"])
            embeddings.append(emb)
            texts.append(chunk["text"])
            metadata.append({"chunk_id": chunk["chunk_id"]})

        if len(embeddings) == 0:
            st.error("No embeddings created. Please upload a valid PDF.")
            st.stop()

        embedding_matrix = np.array(embeddings).astype("float32")
        dim = embedding_matrix.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embedding_matrix)

        st.session_state.index = index
        st.session_state.texts = texts
        st.session_state.metadata = metadata

    st.success("Your document is ready!")

query = st.text_input("💬 Ask something about your PDF")

if query and st.session_state.index is not None:
    with st.spinner("Thinking..."):
        st.session_state.question = query

        query_embedding = get_embedding(query)
        distances, indices = st.session_state.index.search(
            np.array([query_embedding]).astype("float32"),
            3
        )

        context = ""
        citations = []

        for idx in indices[0]:
            context += f"\n\nChunk {st.session_state.metadata[idx]['chunk_id']}:\n"
            context += st.session_state.texts[idx]
            citations.append(st.session_state.metadata[idx]['chunk_id'])

        prompt = f"""
Use ONLY the provided context to answer.
If not found, say you cannot find it.

Context:
{context}

Question:
{query}

Answer:
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        st.session_state.answer = response.choices[0].message.content
        st.session_state.citations = citations

if st.session_state.question:
    st.markdown(f'<div class="chat-bubble-user">🧑 {st.session_state.question}</div>', unsafe_allow_html=True)

if st.session_state.answer:
    st.markdown(f'<div class="chat-bubble-bot">🤖 {st.session_state.answer}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="citation">📚 Cited Chunks: {st.session_state.citations}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)