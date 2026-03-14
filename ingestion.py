# ingestion.py

from pypdf import PdfReader

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ingestion.py

import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()