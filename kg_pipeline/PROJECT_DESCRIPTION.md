# 📘 Knowledge Graph for Text Documents (KG-RAG)

This project is designed to build and query a **knowledge graph** from unstructured documents (like PDFs or plain text) using a combination of **Neo4j**, **Named Entity Recognition (NER)** with **spaCy**, **LLMs (Gemini / LLaMA)**, **Graph RAG (Retrieval-Augmented Generation)**, and **custom evaluation using RAGAS**.

---

## 🚀 Features

- 🔍 Extracts entities using spaCy's NER.
- 🧠 Constructs a knowledge graph using **Neo4j**.
- 🔗 Uses **Graph-RAG** techniques to retrieve context from the graph.
- 🗣 Generates answers using LLMs like **Gemini** or **LLaMA-based models**.
- 📊 Evaluates the quality of generated answers using **RAGAS**.
- ⚙️ Integrates with custom LLM and Embedding components.
- 🌐 Uses `all-MiniLM-L6-v2` for semantic search via embeddings.

---

## 🧱 Tech Stack

| Component         | Technology                           |
|------------------|--------------------------------------|
| NER               | `spaCy`                              |
| Graph DB          | `Neo4j`                              |
| Embeddings        | `all-MiniLM-L6-v2` (via SentenceTransformers) |
| LLMs              | `Gemini`, `LLaMA`                    |
| Retrieval         | `Graph RAG`                          |
| Evaluation        | `RAGAS` with custom LLM + embedding layers |
| Orchestration     | Python scripts                       |

---


## 🗂️ Folder Structure

knowledge-graph-for-txt-docs/ │ ├── kg_pipeline/ │ ├── extract_text.py # Extracts raw text from input documents │ ├── extract_entities.py # Performs NER and prepares data for KG │ ├── graph_builder.py # Builds and loads knowledge graph into Neo4j │ ├── nlp_spacy.py # spaCy utilities for NER │ ├── main.py # 🔹 Entry point for data ingestion (NER + KG) │ ├── delhi_sultnate_test_data.pdf # Sample input document │ ├── kg_pipeline/kg_rag_pipeline/ │ ├── config.py # Configuration for RAG pipeline │ ├── env.example # Environment variable template │ ├── retrieval_pipeline.py # Retrieves context from KG using embeddings │ ├── generator.py # Calls LLM (Gemini/LLaMA) to generate answers │ ├── ranker.py # Ranks retrieved nodes if needed │ ├── ragas_eval.py # 🔸 Evaluates answers using RAGAS metrics │ ├── main.py # 🔹 Entry point to run RAG over KG │ ├── ground_truth.json # Reference answers for evaluation │ ├── results/ # Stores evaluation outputs │ ├── utils/ # Utility functions │ ├── requirements.txt # Python dependencies ├── README.md # 📄 You are here!

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/knowledge-graph-for-txt-docs.git
cd knowledge-graph-for-txt-docs
2. Install Dependencies

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Setup Environment
Create an .env file based on env.example and add your:

Gemini API key

Neo4j credentials

LLM model settings


▶️ How to Run
📥 1. Ingest Documents & Build Knowledge Graph

python kg_pipeline/main.py
This will:
Extract entities using spaCy

Build and store the graph in Neo4j

🧠 2. Run Retrieval-Augmented Generation (Graph-RAG)

python kg_pipeline/kg_rag_pipeline/main.py
This will:
Use Graph-RAG to retrieve relevant subgraphs using embeddings Query Gemini or LLaMA to generate answers

3. Evaluate Answer Quality (RAGAS)

python kg_pipeline/kg_rag_pipeline/ragas_eval.py

This script evaluates generated answers using:
Faithfulness
Answer Relevancy
Context Recall & Precision

⏱️ Note: Includes execution delays to prevent burst throttling on Gemini's free tier. See ragas_eval.py for details.

📌 Notes
Uses all-MiniLM-L6-v2 for semantic retrieval

Evaluation supports custom LLM and embedding layers passed into RAGAS

You can modify ground_truth.json to update reference answers for testing

📬 Contact
For queries or collaborations, reach out at shivam170620@gmail.com