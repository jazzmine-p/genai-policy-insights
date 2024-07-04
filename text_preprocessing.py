import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
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
    
# Text preprocessing
def preprocess_text(all_sections):
    logger.info("Preprocessing text")

    for section in all_sections:
        for text in section:
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'et al', '', text)
            # Remove Markdown-style links ([text](url)) from a section
            text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
            
            # Remove standalone URLs (http/https)
            text = re.sub(r'\b(?:https?://)\S+\b', '', text)

            # Remove URLs starting with www.
            text = re.sub(r'\b(?:www\.)\S+\b', '', text)

    return all_sections