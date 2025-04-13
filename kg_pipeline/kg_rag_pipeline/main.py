# import json
# import concurrent.futures
# from extract_entities import extract_entities_from_query
# from retrieval_pipeline import generate_embedding, Neo4jKG
# from ranker import ranker_and_combine_contexts
# from generator import generate_answer
# from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# from pathlib import Path

# # Load questions
# with open("ground_truth.json", "r") as f:
#     qa_pairs = json.load(f)

# kg = Neo4jKG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# def process_question(item):
#     query = item["question"]
#     ground_truth = item["expected_answers"]

#     try:
#         query_embedding = generate_embedding(query)
#         entities = extract_entities_from_query(query)
#         structural_context = kg.get_structural_context(entities)
#         vector_context = kg.get_vector_context(query_embedding)

#         combined_context = {
#             "structural_context": structural_context,
#             "vector_context": vector_context
#         }

#         answer = generate_answer(query, combined_context)

#         return {
#             "query": query,
#             "ground_truth": ground_truth,
#             "answer": answer,
#             "context": combined_context
#         }

#     except Exception as e:
#         print(f"Error processing query '{query}': {e}")
#         return None

# def main():
#     results = []

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(process_question, q) for q in qa_pairs]
#         for future in concurrent.futures.as_completed(futures):
#             result = future.result()
#             if result:
#                 results.append(result)

#     # Save results
#     Path("results").mkdir(exist_ok=True)
#     with open("results/qa_results.json", "w") as f:
#         json.dump(results, f, indent=2)

#     print("✅ All results saved to results/qa_results.json")

#     kg.close()

# if __name__ == "__main__":
#     main()

import json
import time
from pathlib import Path
from extract_entities import extract_entities_from_query
from retrieval_pipeline import generate_embedding, Neo4jKG
from ranker import ranker_and_combine_contexts
from generator import generate_answer, get_context
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Load questions
with open("ground_truth.json", "r") as f:
    qa_pairs = json.load(f)

kg = Neo4jKG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

def process_question(item):
    query = item["question"]
    ground_truth = item["expected_answers"]

    try:
        start_retrieval = time.time()
        query_embedding = generate_embedding(query)
        entities = extract_entities_from_query(query)
        structural_context = kg.get_structural_context(entities)
        vector_context = kg.get_vector_context(query_embedding)
        retrieval_time = time.time() - start_retrieval

        combined_context = {
            "structural_context": structural_context,
            "vector_context": vector_context
        }

        context = get_context(combined_context)

        start_generation = time.time()
        answer = generate_answer(query, context)
        if hasattr(answer, "content"):
            answer = answer.content
        else:
            answer = str(answer)
        generation_time = time.time() - start_generation

        return {
            "query": query,
            "ground_truth": ground_truth,
            "answer": answer,
            "context": context,
            "timing": {
                "retrieval_time_sec": round(retrieval_time, 2),
                "generation_time_sec": round(generation_time, 2)
            }
        }

    except Exception as e:
        print(f"❌ Error processing query '{query}': {e}")
        return None

def main():
    results = []

    for idx, q in enumerate(qa_pairs):
        print(f"⚙️ Processing {idx + 1}/{len(qa_pairs)}: {q['question']}")
        result = process_question(q)
        if result:
            results.append(result)

        # Enforce 2 requests per minute (1 request every 30 sec)
        if idx != len(qa_pairs) - 1:
            print("⏳ Sleeping for 30 seconds to respect token limits...")
            time.sleep(30)

    # Save results

    print("results upon running iterations -->  " ,results)
    Path("results").mkdir(exist_ok=True)
    with open("results/qa_results.json", "w") as f:
        json.dump(results, f, indent=2)


    print("✅ All results saved to results/qa_results.json")
    kg.close()

if __name__ == "__main__":
    main()
