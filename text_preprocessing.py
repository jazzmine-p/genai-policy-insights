import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import spacy
import re
import logging

logger = logging.getLogger(__name__)

# Delete unwanted sections
def filter_sections(all_sections):
    logger.info("Filtering sections")
    
    unwanted_sections_header = ["references", "appendix", "glossary", 'table of contents', 'acknowledgments', 'disclosure statement', 'author', 'contact', 'executive summary', 'introduction']
    unwanted_sections = [section for section in all_sections if any(text in re.search(r'#{1,6} (.*)', section).group(1)  for text in unwanted_sections_header)]
    filtered_sections = [section for section in all_sections if section not in unwanted_sections]

    return filtered_sections


nlp = spacy.load("en_core_web_sm")
def remove_names(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "DATE"]]

    logging.info(f"Entities: {entities}")

    for entity in entities:
        text = text.replace(entity, "")
    return text

# Text preprocessing
def preprocess_text(all_sections):
    logger.info("Preprocessing text")

    preprocessed_sections = []

    phrase_mapping = {
    "genai": "generative AI",
    "gen ai": "generative AI",
    "generative ai": "generative AI",
    "generative artificial intelligence": "generative AI",
    }

    for text in all_sections:
        # Remove names
        text = remove_names(text)
        # Remove numbers
        processed_text = re.sub(r'\d+', '', text)
        # Remove "et al"
        processed_text = re.sub(r'et al', '', processed_text)
        # Remove page divider
        processed_text = re.sub(r'\n\n\n-+\n\n', ' ', processed_text)
        # Normalize white spaces
        processed_text = re.sub(r'\s+', ' ', processed_text)
        # Remove non-word characters and white space
        #processed_text = re.sub(r'\W', ' ', processed_text)
        # Remove Markdown-style links
        processed_text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', processed_text)
        # Remove standalone URLs (http/https)
        processed_text = re.sub(r'\b(?:https?://)\S+\b', '', processed_text)
        # Remove URLs starting with www.
        processed_text = re.sub(r'\b(?:www\.)\S+\b', '', processed_text)
        # Consolidate phrases
        for key, value in phrase_mapping.items():
            processed_text = re.sub(r'\b' + re.escape(key) + r'\b', value, processed_text, flags=re.IGNORECASE)

        preprocessed_sections.append(processed_text)

    return preprocessed_sections