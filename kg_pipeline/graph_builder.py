# graph_builder.py

# Given subject object and predicate , we can create relationships between them.

# Node details 

# Storing name, chunk, page, embedding 

# <id>: 4:b1800785-618f-49bd-ba40-bf471a8e7a0c:382
# name: "1305"
# chunk: "performed Jauhar. This episode has been mentioned in the book 
# Padmavat written by Jayasi. 
# • 
# Malwa and others: Under the able leadership of Ain-ul-Mulk, the Khalji 
# army captured Malwa in 1305. Ujjain, Mandu, Chanderi and Dhar were 
# also annexed. After the annexation of Jalore in 1311, Alauddin Khalji 
# became the master of north India after having captured large parts of the 
# Rajputana. 
# 3. 
# Conquest of Deccan and the far South: The conquest of Deccan and"
# page: 8
# embedding: [-0.09176092594861984, 0.07729890942573547, -0.025281433016061783, -0.007341655436903238, -0.09600866585969925, ..]


# Relationship details 

# <id>: 5:b1800785-618f-49bd-ba40-bf471a8e7a0c:6917608192478282075
# source: "• 
# He was a pacifist to a great extent and tried to soften some of the harsher aspects of 
# Balban’s rule.  
# • 
# He was the first ruler of the Delhi Sultanate who was secular to a great extent and 
# denied India to be an Islamic state as the majority of the population was Hindu. To him, 
# a state should be based on the generous support of its people.  
# • 
# He adopted the policy of tolerance and avoided harsh punishments. However, his reign"
# embedding: [0.03172924369573593, 0.10186281055212021, -0.051756057888269424, ..]

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


# Load model
model = SentenceTransformer("all-MiniLM-L6-v2") # 384 dimension embedding vector


def generate_embedding(text):
    return model.encode(text).tolist()

class Neo4jKG:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    @staticmethod
    def _sanitize_predicate(predicate):
        # Convert to uppercase
        predicate = predicate.upper()
        # Replace spaces with underscores
        predicate = predicate.replace(' ', '_')
        # Remove invalid characters (anything that's not a letter, number, or underscore)
        predicate = ''.join(char for char in predicate if char.isalnum() or char == '_')
        # Ensure the predicate is not empty and doesn't start with a non-letter character
        if not predicate or not predicate[0].isalpha():
            raise ValueError(f"Invalid predicate name after sanitization: '{predicate}'")
        return predicate

    def insert_triple(self, subject, predicate, obj, chunk_text, page_no):
            # Sanitize the predicate to create a valid relationship type
        try:
            sanitized_predicate = self._sanitize_predicate(predicate)
        except ValueError:
            return

        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, subject, sanitized_predicate, obj, chunk_text, page_no)

    @staticmethod
    def _create_relationship(tx, subject, predicate, obj, chunk_text, page_num = -1):
        # print(predicate.upper().replace(' ', '_'))
        embedding = generate_embedding(chunk_text)

        query = (
        f"""
        MERGE (a:Entity {{name: $subject}})
        ON CREATE SET a.chunk = $chunk_text, a.page = $page_num, a.embedding = $embedding 

        MERGE (b:Entity {{name: $object}})
        ON CREATE SET b.chunk = $chunk_text, b.page = $page_num, b.embedding = $embedding

        MERGE (a)-[r:{predicate}]->(b)
        ON CREATE SET r.source = $chunk_text, r.embedding = $embedding
        """)

        tx.run(query, subject=subject, object=obj, chunk_text=chunk_text, 
               page_num=page_num, embedding=embedding)
