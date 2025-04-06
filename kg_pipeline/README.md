# knowledge-graph-for-txt-docs


### Text extarction from pdf and txt documemt 

1. use pymupdf for this task 

### NLP with spacy 

1. we will be using spacy for named enity recognition 

Example 1: General
Input Text:

"Apple Inc. was founded by Steve Jobs in California in 1976."

NER Output:

Apple Inc. → ORG (Organization)

Steve Jobs → PERSON

California → GPE (Geo-Political Entity / Location)

1976 → DATE

Example 2: Medical Domain
Input Text:

"Paracetamol is often prescribed to reduce fever and relieve mild pain."

NER Output:

Paracetamol → DRUG (domain-specific model)

fever → SYMPTOM (if using medical NER models like ScispaCy or Med7)

mild pain → SYMPTOM

🧠 Note: For healthcare, use models like en_core_sci_md or med7 for better results.


### Triplet formation 

convert sentences into (subjects, relation, objects).

### Neo4j Storage 

Use neo4j python driver to store entities 


