
import numpy as np
from retrieval_pipeline import generate_embedding

def ranker_and_combine_contexts(structural_context, vector_context, query_embedding, ranked_top_k=10):
    
    # Process structural context
    ranked_structural = []
    for path in structural_context:
        # Extract all text from the path
        path_text = ""
        for node in path["nodes"]:
            if node["chunk"]:
                path_text += node["chunk"] + " "
        
        # Calculate similarity score for the whole path
        if path_text:
            path_embedding = generate_embedding(path_text)
            # Cosine similarity
            path_similarity = np.dot(path_embedding, query_embedding) / (np.linalg.norm(path_embedding) * np.linalg.norm(query_embedding))
            
            ranked_structural.append({
                "path": path,
                "similarity": float(path_similarity),
                "source": "structural"
            })
    
    # Process vector context
    ranked_vector = []
    for node in vector_context:
        ranked_vector.append({
            "node": node,
            "similarity": node["similarity"],
            "source": "vector"
        })
    
    # Combine and sort all contexts
    all_contexts = []
    all_contexts.extend(ranked_structural)
    all_contexts.extend(ranked_vector)
    
    # Sort by similarity score
    all_contexts.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top combined contexts (adjust as needed)
    return all_contexts[:ranked_top_k]