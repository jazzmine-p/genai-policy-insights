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

nlp = spacy.load("en_core_web_sm")

def remove_names(text):
    doc = nlp(text)

    ent_dict = {ent.text: ent.label_ for ent in doc.ents if ent.label_ in entity_labels}
    entities = ent_dict.keys()

    for entity in entities:
        text = text.replace(entity, "")
    return text, ent_dict

# Text preprocessing
def preprocess_text(docs):
    logger.info("Preprocessing text")

    preprocessed_sections = []
    all_entities = []

    for text in docs:
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
        # Remove names
        processed_text, ent = remove_names(processed_text)
        all_entities.extend(ent.items())
        # Remove numbers
        processed_text = re.sub(r'\d+', '', processed_text)
        # Remove "et al"
        processed_text = re.sub(r'et al', '', processed_text)
        # Remove page divider
        processed_text = re.sub(r'\n\n\n-+\n\n', ' ', processed_text)
        # Normalize white spaces
        processed_text = re.sub(r'\s+', ' ', processed_text)
        # Remove non-word characters and white space
        processed_text = re.sub(r'\W', ' ', processed_text)

        preprocessed_sections.append(processed_text)

        # Convert the accumulated entity data to a DataFrame
        ent_df = pd.DataFrame(all_entities, columns=['Entity', 'Label'])
    
        # Save the DataFrame to a CSV file
        ent_df.to_csv(f'{log_dir}/removed_entities.csv', index=False)

    return preprocessed_sections