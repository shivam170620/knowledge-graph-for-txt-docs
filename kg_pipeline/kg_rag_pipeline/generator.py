from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
from config import GROQ_API_KEY, GROQ_MODEL_NAME

def get_context(context):
    structural_text = ""
    if context["structural_context"]:
        structural_text = "Graph Structural Context:\n"
        for i, path in enumerate(context["structural_context"]):
            structural_text += f"Path {i+1}:\n"
            for node in path["nodes"]:
                structural_text += f"- Node: {node['name']}\n  Content: {node['chunk'][:200]}...\n"
            structural_text += "\n"
    
    # Format vector context
    vector_text = ""
    if context["vector_context"]:
        vector_text = "Vector Similarity Context:\n"
        for i, node in enumerate(context["vector_context"]):
            vector_text += f"Document {i+1} (Similarity: {node['similarity']:.4f}):\n"
            vector_text += f"Content: {node['chunk']}\n"
            vector_text += f"Page: {node['page']}\n\n"
    
    # Combine contexts
    combined_context = structural_text + "\n" + vector_text
    return combined_context

def generate_answer(query , context, ):

    """
        Generate an answer using retrieved context
        """
    
    # Generate answer using OpenAI
    system_prompt = """
    You are an expert researcher with access to a knowledge graph constructed from a collection of documents.
    Answer the user's question based on the structured and unstructured context available in the graph.
    Use relationships, entities, and semantic connections from the knowledge graph to infer meaningful answers.
    Cite specific information or nodes when possible.
    If the answer cannot be determined from the available context, clearly state so without making unsupported assumptions.
    """
    
    messages = [
        {"role" : "system",  "content" : system_prompt},
        { "role" : "user" , "content" : f"Question: {query}\n\nContext:\n{context}"}
    ]

    # print("messages ------------->" , messages)

    llm = ChatGroq(temperature=0.5, groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL_NAME, max_retries=2)
    response = llm.invoke(messages)

    print("response generated from groq --", response)
    
    return response