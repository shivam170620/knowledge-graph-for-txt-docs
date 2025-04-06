# === Extract Text from PDF ===

import fitz
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# we can use pdfplumber for extracting texts from pdf, also we can use diff kind of files, I am considering 
# only pdf file for now.

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return [(page_num + 1, page.get_text("text")) for page_num, page in enumerate(doc)]
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return []


def chunk_pdf_text(pdf_path):
    page_texts = extract_text_from_pdf(pdf_path)

    if not page_texts:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    chunks_with_pages = []

    for page_number, text in page_texts:
        if not text.strip():
            continue

        page_chunks = text_splitter.split_text(text)
        for chunk in page_chunks:
            chunks_with_pages.append({
                "page_number": page_number if page_number else -1,
                "chunk_text": chunk
            })

    return chunks_with_pages

