

from extract_entities import extract_entities_from_query
from retrieval_pipeline import generate_embedding
from ranker import ranker_and_combine_contexts
from generator import generate_answer
from retrieval_pipeline import Neo4jKG
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def main():
    """
    Main entry point for the QA system
    """

    kg = Neo4jKG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    # 1. Extract query embedding
    query = "Name the five dynasties that ruled the Delhi Sultanate ?"
    query_embedding = generate_embedding(query)
    
    # 2. Extract potential entities from query using spaCy
    entities = extract_entities_from_query(query)
    print(f"Extracted entities: {entities}")
    
    # 3. Get structural context from graph relationships
    structural_context = kg.get_structural_context(entities)
    
    # 4. Get vector similarity context
    vector_context = kg.get_vector_context(query_embedding)

    # ranked_contexts = ranker_and_combine_contexts(structural_context,vector_context, query_embedding)
    
    # 5. Combine contexts
    combined_context = {
        "query": query,
        "structural_context": structural_context,
        "vector_context": vector_context,
        "extracted_entities": entities
    }

    # print(combined_context)
    
    # 6. Generate answer
    answer = generate_answer(query, combined_context)
    
    print(f"Question: {query}")
    print(f"Entities extracted: {combined_context['extracted_entities']}")
    print(f"Answer: {answer}")

    kg.close()

if __name__ == "__main__":

    main()