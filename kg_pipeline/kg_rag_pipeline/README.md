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






questions --- 



{
      "question": "How did the Sayyid dynasty come to an end?",
      "expected_answers": [
        "Alam Shah was overthrown by Bahlol Lodhi",
        "Hamid Khan invited Lodhi to attack the Sultan",
        "Marked the end of Sayyid rule in 1451"
      ]
    },
    {
      "question": "What were the religious policies of Sikandar Lodhi?",
      "expected_answers": [
        "Reimposed Jizya on non-Muslims",
        "Was intolerant towards other religions",
        "Tortured poet Kabirdas"
      ]
    },
    {
      "question": "Who was responsible for founding the city of Agra and when?",
      "expected_answers": [
        "Sikander Lodhi",
        "Founded Agra in 1504"
      ]
    },
    {
      "question": "What led to the downfall of Ibrahim Lodhi and the end of the Delhi Sultanate?",
      "expected_answers": [
        "Defeated by Babur in the First Battle of Panipat (1526)",
        "Internal dissent and weak leadership",
        "Marked the beginning of Mughal rule"
      ]
    },
    {
      "question": "Explain the cultural contributions of Alauddin Khalji.",
      "expected_answers": [
        "Patronized poets like Amir Khusrau",
        "Built Alai Darwaza and Siri fort",
        "Called Sikander-i-Azam and supported literature"
      ]
    },
    {
      "question": "What was the significance of the Iqta system introduced by Iltutmish?",
      "expected_answers": [
        "Allowed nobles to collect revenue in exchange for military service",
        "Provided a salary-based land allocation",
        "Foundation of feudal administrative structure in India"
      ]
    },
    {
      "question": "Summarize the achievements of the Mamluk dynasty in one paragraph.",
      "expected_answers": [
        "The Mamluk dynasty laid the foundation of the Delhi Sultanate with rulers like Qutb-ud-din Aibak and Iltutmish. They established political stability, introduced key administrative systems like Iqtas, coinage, and architectural contributions like the Qutub Minar. Their reign witnessed Mongol threats, court politics, and the emergence of female leadership under Razia Sultan."
      ]
    },
  
    {
      "question": "Name one dynasty that ruled between 1206–1290.",
      "expected_answers": ["Mamluk dynasty"]
    },
    {
      "question": "Which dynasty was ruling during 1414–1451?",
      "expected_answers": ["Sayyid dynasty"]
    },
    {
      "question": "Describe the significance of the Delhi Sultanate in shaping the culture and geography of India.",
      "expected_answers": ["Unified large parts of India", "Introduced Persian culture", "Architectural developments", "Established Indo-Islamic traditions"]
    },
    {
      "question": "What was the time period during which the Delhi Sultanate existed?",
      "expected_answers": ["1206–1526", "13th to 16th century"]
    },
    {
      "question": "When did the Khalji dynasty rule the Delhi Sultanate?",
      "expected_answers": ["1290–1320"]
    },
    {
      "question": "Explain the administrative reforms introduced by Alauddin Khalji.",
      "expected_answers": ["Market control", "Efficient spy system", "Revenue reforms"]
    },
    {
      "question": "During whose reign was the token currency introduced and why did it fail?",
      "expected_answers": ["Muhammad bin Tughlaq", "Rampant forgery", "Loss of public trust"]
    },
    {
      "question": "What dynasties ruled the Delhi Sultanate in chronological order?",
      "expected_answers": ["Mamluk", "Khalji", "Tughlaq", "Sayyid", "Lodi"]
    },
    {
      "question": "In what year was the First Battle of Panipat fought, and what was its outcome?",
      "expected_answers": ["1526", "End of Delhi Sultanate", "Babur defeated Ibrahim Lodi", "Start of Mughal Empire"]
    },
    {
      "question": "Describe the architectural contributions of the Tughlaq dynasty.",
      "expected_answers": ["Tughlaqabad Fort", "Simple and strong architectural style"]
    },
    {
      "question": "When did Balban rule and what was his policy toward the nobles?",
      "expected_answers": ["1266–1287", "Crushed nobility", "Strengthened monarchy"]
    },
    {
      "question": "Explain the role of Iltutmish in consolidating the Delhi Sultanate.",
      "expected_answers": ["Introduced Iqta system", "Completed Qutub Minar", "Strengthened administration"]
    }, {
      "question": "Name the five dynasties that ruled the Delhi Sultanate.",
      "expected_answers": ["Mamluk", "Khalji", "Tughlaq", "Sayyid", "Lodi"]
    },
    {
      "question": "Which areas did the Delhi Sultanate cover?",
      "expected_answers": ["India", "Pakistan", "Bangladesh", "Nepal"]
    },
    {
      "question": "What is the modern relevance of the Delhi Sultanate?",
      "expected_answers": ["Cultural influence", "Architectural legacy"]
    },
    {
      "question": "Which dynasty ruled from 1320 to 1414?",
      "expected_answers": ["Tughlaq dynasty"]
    },
    {
      "question": "What was the last dynasty of the Delhi Sultanate?",
      "expected_answers": ["Lodi dynasty"]
    },
    {
      "question": "What caused the failure of Muhammad bin Tughlaq’s token currency policy?",
      "expected_answers": [
        "People forged copper coins easily",
        "Loss of trust in currency led to treasury loss",
        "Policy was reversed and gold/silver currency restored"
      ]
    },
    {
      "question": "Why did Muhammad bin Tughlaq shift his capital to Daulatabad and what went wrong?",
      "expected_answers": [
        "Aimed to control the south better",
        "Forced population migration caused suffering",
        "Water shortage led to reversal of decision"
      ]
    },
    {
      "question": "What were the administrative contributions of Firoz Shah Tughlaq?",
      "expected_answers": [
        "Built canals and gardens",
        "Promoted scholars and translated Sanskrit works",
        "Revived hereditary Iqta system"
      ]
    }