# Running RAG pipeline on Knowledge graphs where our data is stored 

We need to employ graph rag for the same. 

Also knowledge graph has 2 components of information , structured data and unstructured data 

##  Approach 1 

We can use cypher query for the find the relevant chunks and then apply vector similarity search 

This is what we called adavnced rag (GRAPH RAG)

uses structured knowledge in the form of KG.
enhances semantic understanding using text embeddings stored in the graph 
Retrieve relevant subgraphs using similarity search

### Subgraph retrival top_k 

Fetch semantically similar nodes and expand to neighbours 
prompt construction : format subgraph and form full context which should be passed to LLM

## Similarity search against all nodes is computationally expensive

performing a similarity search against all nodes with embeddings can become expensive as your graph grows.

🔍 Why the Time Complexity Can Be High:
In this Cypher query:

MATCH (n)
WHERE n.embedding IS NOT NULL
WITH n, gds.similarity.cosine(n.embedding, queryVector) AS similarity
You're:

Scanning every node n in the graph

Computing cosine similarity with each node’s embedding

Doing this in-memory for each query

So the time complexity is roughly:

O(N * D)
Where:

N = number of nodes with embeddings

D = dimensionality of the embedding vector (usually 384, 768, or 1024)

This is linear in the number of nodes, but still expensive for large graphs (millions of nodes).

Solution :- 

Use HNSW Vector Indexing (available in Neo4j GDS 2.4+):
Neo4j supports approximate nearest neighbor (ANN) search using HNSW (Hierarchical Navigable Small World graphs):


# Cosine similarity is not available through apoc functions , neither the dot product .. 

Using numpy for this purpose 

