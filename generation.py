from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
def build_prompt(query, retrieved_chunks):
    context = ""
    for chunk in retrieved_chunks:
        context += f"""
Source: {chunk['metadata']['document']} | Chunk: {chunk['metadata']['chunk_id']}
{chunk['text']}
"""
    
    prompt = f"""
You are a helpful assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say:
"I could not find the answer in the provided documents."

Provide citations in this format:
(Document name, Chunk ID)

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt


def generate_answer(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content