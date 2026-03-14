# chunking.py

def chunk_text(text: str, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk
        })
        
        start += chunk_size - overlap
        chunk_id += 1
    
    return chunks