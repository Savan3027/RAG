from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)