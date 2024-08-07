import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import yaml
import spacy
import re
import logging
from modules.config.constants import log_dir

logger = logging.getLogger(__name__)
with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
entity_labels = config['unwanted_entity_labels']
unwanted_words = config['unwanted_words']

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2800000

def remove_ner(text):
    doc = nlp(text)

    lemmatized_tokens = []
    removed_entities = {}

    for token in doc:
        if token.ent_type_ in entity_labels:
            if token.ent_type_ not in removed_entities:
                removed_entities[token.ent_type_] = []
            removed_entities[token.ent_type_].append(token.text)
        else:
            lemmatized_tokens.append(token.lemma_)

    # Construct the final text while handling punctuation and whitespace
    lemmatized_text = ''
    for i, token in enumerate(doc):
        if token.ent_type_ not in entity_labels:
            lemmatized_text += token.lemma_

            # Add space only if the next token is not punctuation
            if i + 1 < len(doc) and not doc[i + 1].is_punct:
                lemmatized_text += ' '

    return lemmatized_text.strip(), removed_entities

def preprocess_text(text):
    # Lowercase the markdown text to make the regex case-insensitive
    text = text.lower()
    # NER and lemmatization
    processed_text, ent = remove_ner(text)
    # Remove numbers
    processed_text = re.sub(r'\d+', '', processed_text)
    # Remove words
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, unwanted_words)))
    processed_text = re.sub(pattern, '', processed_text)
    # Remove page divider
    #processed_text = re.sub(r'\n\n\n-+\n\n', ' ', processed_text)
    # Remove underscores
    processed_text = re.sub(r'_', ' ', processed_text)
    # Normalize white spaces
    #processed_text = re.sub(r'\s+', ' ', processed_text)
    # Remove non-word characters and white space
    #processed_text = re.sub(r'\W', ' ', processed_text)
    #logging.info("Remove links, consolidating phrases, NER, remove underscores")
    return processed_text