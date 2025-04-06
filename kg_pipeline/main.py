# main.py

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, PDF_PATH
from extract_text import chunk_pdf_text
from nlp_spacy import extract_triplets
from graph_builder import Neo4jKG

def main():
    print("📄 Extracting text from PDF...")
    chunks = chunk_pdf_text(PDF_PATH)

    if not chunks:
        print("⚠️ No text extracted or chunking failed.")
        return

    print("\n🧠 Extracting triples using spaCy...")

    kg = Neo4jKG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # print("Chunks" , chunks)
    # print(isinstance(chunks, dict) , " --", type(chunks))

   
    for chunk in chunks:
        if isinstance(chunk, dict):
            page_number = chunk["page_number"]
            chunk_text = chunk["chunk_text"]
            print(f"Page {page_number}: {chunk_text[:50]}...")
        else:
            print("⚠️ Unexpected chunk format:", chunk)


        triples = extract_triplets(chunk_text)
        
        if not triples:
            print("No triples found in this chunk.")
            continue

        for subj, pred, obj in triples:
            kg.insert_triple(subj, pred, obj, chunk_text,page_number)

    kg.close()

    print("\n✅ Knowledge Graph building complete!")

if __name__ == "__main__":
    main()
