



# Function to extract all the entities using spacy 

import spacy 
import json


nlp = spacy.load("en_core_web_sm")

def extract_entities_from_query(query: str):

    doc = nlp(query) # running small spacy model for ner
    entities = []

    for entity in doc.ents:
        entities.append(entity.text)

    # extract noun phrases 

    for chunk in doc.noun_chunks: 
        if len(chunk.text.split()) > 1:  # Multi-word phrases are more likely to be specific entities
                entities.append(chunk.text)

    # if entities are not being extracted, extract the import words from part of speech tagging 
    filtered_entities = [entity for entity in entities if len(entity) > 2]

    print("Filtered Entities --:", filtered_entities)

    if not filtered_entities:
        important_pos = ['PROPN', 'NOUN', 'ADJ']
        for token in doc:
            if token.pos_ in important_pos and not token.is_stop:
                filtered_entities.append(token.text)
         
    return list(set(filtered_entities))
         





