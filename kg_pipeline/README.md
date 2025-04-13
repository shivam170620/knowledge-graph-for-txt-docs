# 📘 Knowledge Graph for Text Documents (KG-RAG)

A pipeline to extract, organize, and query knowledge from unstructured text using Neo4j, spaCy, Graph-RAG, and LLMs.

---

## 🚀 Features

- 🔍 Entity Extraction using `spaCy`
- 🧠 Knowledge Graph built in `Neo4j`
- 🔗 Graph-RAG for retrieval and context building
- 🤖 Answer Generation using `Gemini` / `LLaMA`
- 📊 Performance Evaluation via `RAGAS`
- 🌐 Semantic similarity via `all-MiniLM-L6-v2`

---

## 📊 Tech Stack

| Component     | Technology                        |
|---------------|-----------------------------------|
| NER           | spaCy                             |
| Graph DB      | Neo4j                             |
| Embeddings    | all-MiniLM-L6-v2 (SentenceTransformers) |
| LLMs          | Gemini, LLaMA                     |
| Retrieval     | Graph-RAG                         |
| Evaluation    | RAGAS                             |
| Language      | Python                            |

---

## 🗂 Folder Structure

knowledge-graph-for-txt-docs/
├── kg_pipeline/
│   ├── extract_text.py           # Extracts text from documents
│   ├── extract_entities.py       # spaCy NER and triplet generation
│   ├── graph_builder.py          # Builds knowledge graph in Neo4j
│   ├── nlp_spacy.py              # spaCy setup and helper functions
│   ├── main.py                   # Entry point to build KG
│   └── delhi_sultnate_test_data.pdf
│
├── kg_pipeline/kg_rag_pipeline/
│   ├── config.py                 # API keys, model names, Neo4j settings
│   ├── env.example               # Template for environment variables
│   ├── retrieval_pipeline.py     # Retrieves context using embeddings
│   ├── generator.py              # Generates answers using LLMs
│   ├── ranker.py                 # (Optional) Ranks graph nodes
│   ├── ragas_eval.py             # RAGAS-based answer evaluation
│   ├── main.py                   # Graph-RAG entry point
│   ├── ground_truth.json         # Reference answers
│   └── results/                  # Stores evaluation outputs
│
├── utils/                        # Embeddings, LLMs, ANN logic
├── requirements.txt              # Python dependencies
└── README.md

## Setup Instructions
1. Clone the Repository

git clone https://github.com/your-username/knowledge-graph-for-txt-docs.git
cd knowledge-graph-for-txt-docs
2. Create Environment & Install Dependencies

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt


3. Configure Environment
Create a .env file:

NEO4J_URI=neo4j+s://your-uri.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

GROQ_API_KEY=your-groq-key
GROQ_MODEL_NAME=llama-3.1-8b-instant
GEMINI_API_KEY=your-gemini-key


▶️ How to Run
1. Build the Knowledge Graph
bash
Copy
Edit
python kg_pipeline/main.py
This will:

Extract text

Apply spaCy NER

Store triplets in Neo4j

2. Run Graph-RAG (Retriever + Generator)
bash
Copy
Edit
python kg_pipeline/kg_rag_pipeline/main.py
This will:

Embed user query

Perform ANN on graph node embeddings

Generate answer using LLM

3. Evaluate Answers with RAGAS

python kg_pipeline/kg_rag_pipeline/ragas_eval.py
Evaluates:

Faithfulness
Context Precision/Recall
Answer Relevancy
Answer Similarity

🔍 Named Entity Recognition (NER) with spaCy
Example 1: General
Input:
"Apple Inc. was founded by Steve Jobs in California in 1976."

Output:

Apple Inc. → ORG

Steve Jobs → PERSON

California → GPE

1976 → DATE

Example 2: Medical
Input:
"Paracetamol is often prescribed to reduce fever and relieve mild pain."

Output:

Paracetamol → DRUG

fever → SYMPTOM

mild pain → SYMPTOM

💡 Use domain-specific models like en_core_sci_md, med7 for healthcare use-cases.

🧰 Triplet Formation
From sentence:

"Steve Jobs founded Apple in 1976."

Creates:
(Subject: Steve Jobs, Predicate: founded, Object: Apple)
🔹 Neo4j Graph Storage
Neo4j setup:

from neo4j import GraphDatabase

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_node(tx, label, properties):
    query = f"CREATE (n:{label} $props)"
    tx.run(query, props=properties)
⚠️ Similarity Search Optimization
Challenge
Cosine similarity across all node embeddings → O(N * D)

Solution
Use HNSW (Hierarchical Navigable Small World Graph) for Approximate Nearest Neighbor Search

Benefits:

Sub-linear retrieval

Efficient scaling

📩 Contact
Shivam Kumar
📧 Email: shivam170620@gmail.com


Thanks to Chatgpt and doc support in assisting me in building this.