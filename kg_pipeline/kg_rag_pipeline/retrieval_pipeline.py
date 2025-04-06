from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from typing import List
from neo4j.work.query import Query
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2") 

def generate_embedding(text):
    return model.encode(text).tolist()

# Geberate relevant graph paths starting from identified entities 
# Return paths with nodes and relationships 

class Neo4jKG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_structural_context(self , entities: List[str] ,max_hops: int = 2,top_k_paths: int = 5,
                                vector_similarity_threshold: float = 0.7 ):
        
        structural_context = []
        with self.driver.session() as session:
            for entity in entities:

                query = Query(text = """
                        MATCH (n) 
                        WHERE 
                        n.name CONTAINS $entity OR 
                        n.chunk CONTAINS $entity OR
                        apoc.text.fuzzyMatch(n.chunk, $entity) > $vector_similarity_threshold
                        WITH n
                        MATCH path = (n)-[r*1..2]-(connected)
                        RETURN path
                        LIMIT $limit
                        """)
                # max_hops value is hardcoded as 2, as in the apoc (awesome preocedures in cypher)
                # apoc.run is not available neo4j aura free instance 

                # CALL apoc.cypher.run(
                #     '
                #     MATCH (n)-[r*1..$max_hops]-(m)
                #     RETURN n, r, m
                #     ',
                #     { max_hops: 3 }
                #     ) YIELD value
                #     RETURN value

                # Above query is dynamic cypher query  reqquire apoc access


                
                print(isinstance(query, str), "type", type(query))
                result = session.run(query , entity=entity, 
                                    limit = top_k_paths, vector_similarity_threshold=
                                     vector_similarity_threshold)
                
                print(f"Result retrieved upon runnung the : {query} \n result : {result}")

                # Storing all nodes and relationships data 
                for record in result:
                    path = record["path"]
                    path_data = {
                        "nodes": [],
                        "relationships": []
                    }
                    
                    # Extract node data
                    for node in path.nodes:
                        node_data = {
                            "id": node.id,
                            "name": node.get("name", ""),
                            "chunk": node.get("chunk", ""),
                            "page": node.get("page", "")
                        }
                        path_data["nodes"].append(node_data)
                    
                    # Extract relationship data
                    for rel in path.relationships:
                        rel_data = {
                            "id": rel.id,
                            "type": rel.type,
                            "source": rel.start_node.id,
                            "target": rel.end_node.id,
                            "properties": dict(rel)
                        }
                        path_data["relationships"].append(rel_data)
                    
                    structural_context.append(path_data)
                
                # print("structural_context --: ", structural_context)
            
        return structural_context

    def get_vector_context(self, query_embedding, top_k_vectors: int = 5,
                       vector_similarity_threshold: float = 0.7):

        # We are doing retrieval of nodes and contents in it on the basis of semantically 
        # i.e vector similarity with query

        # Return top_k most similar nodes with their context content

        query_embedding_np = np.array(query_embedding)
        
        with self.driver.session() as session:
            # Fetch all nodes with embeddings
            result = session.run(
                """
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                RETURN n.name AS name, n.chunk AS chunk, n.page AS page, n.embedding AS embedding
                """
            )
            
            # Calculate similarity in Python
            all_nodes = []
            for record in result:
                node_embedding = np.array(record["embedding"])
                # Calculate cosine similarity
                similarity = np.dot(query_embedding_np, node_embedding) / (
                    np.linalg.norm(query_embedding_np) * np.linalg.norm(node_embedding)
                )
                
                if similarity > vector_similarity_threshold:
                    all_nodes.append({
                        "name": record["name"],
                        "chunk": record["chunk"],
                        "page": record["page"],
                        "similarity": float(similarity)
                    })
            
            # Sort by similarity and take top k
            all_nodes.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Extract the context (chunks) from the top k nodes
            return all_nodes[:top_k_vectors]
            
        

# Explanation of the cypher query for finding the nodes, relationships and connected nodes i.e subgraph

# 1. MATCH (n)
# This starts by matching all nodes in the Neo4j graph and binding them to the variable n.

# 2. WHERE ...
# This filters the matched nodes based on how relevant they are to the input search query ($entity):

# n.name CONTAINS $entity: Node's name property contains the input string.

# n.chunk CONTAINS $entity: Node's chunk (text content) contains the input string.

# apoc.text.fuzzyMatch(n.chunk, $entity) > 0.7: Using APOC library's fuzzy string matching, it checks for semantic similarity between chunk and input query — allowing for fuzzy or partial matches.

# ✅ This part is for searching semantically relevant nodes.

# 3. WITH n
# Passes only the matching nodes (n) forward to the next part of the query.

# 4. MATCH path = (n)-[r*1..${max_hops}]-(connected)
# This matches paths starting from the relevant node n, with relationships r of length 1 to max_hops (e.g., 1–2 or 1–3 hops).

# -[r*1..${max_hops}]- is a variable-length relationship pattern, matching any node connected within the defined number of hops.

# ✅ This part builds a subgraph around the matched entity node.

# 5. RETURN path
# Returns the entire path (node + relationships + connected nodes).

# 6. LIMIT $limit
# Limits the number of paths returned to avoid excessive data.


