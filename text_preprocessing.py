import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import yaml
import spacy
import re
import logging
from constants import log_dir

logger = logging.getLogger(__name__)
with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
entity_labels = config['entity_labels']
words_to_remove = config['words_to_remove']

nlp = spacy.load("en_core_web_sm")

def remove_ner(text):
    doc = nlp(text)

    # Create a dictionary to store the entities and their labels
    ent_dict = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in entity_labels}
    
    # Extract and remove the spans of the entities to be removed
    spans = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ in entity_labels]
    spans = sorted(spans, reverse=True)
    for start, end in spans:
        text = text[:start] + text[end:]

    return text, ent_dict

def preprocess_text(text):
    # Remove Markdown-style links
    processed_text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # Remove standalone URLs (http/https)
    processed_text = re.sub(r'\b(?:https?://)\S+\b', '', processed_text)
    # Remove URLs starting with www.
    processed_text = re.sub(r'\b(?:www\.)\S+\b', '', processed_text)
    # Consolidate phrases
    phrase_mapping = config['phrase_mapping']
    for key, value in phrase_mapping.items():
        processed_text = re.sub(r'\b' + re.escape(key) + r'\b', value, processed_text, flags=re.IGNORECASE)
    # NER
    processed_text, ent = remove_ner(processed_text)
    # Remove predefined words
    #pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, words_to_remove)))
    #processed_text = re.sub(pattern, '', processed_text)
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