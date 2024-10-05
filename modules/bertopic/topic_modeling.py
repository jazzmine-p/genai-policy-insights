import os
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
import yaml
import openai
import logging
from modules.bertopic.text_preprocessing import preprocess_text
from modules.config.constants import *

logger = logging.getLogger(__name__)

with open(config_bertopic_dir, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Access hyperparameters for each model
embedding_config = config['embedding_model']['minilm-sm']
umap_config = config['umap_model']
hdbscan_config = config['hdbscan_model']
vectorizer_config = config['vectorizer_model']
mmr_config = config['mmr_model']
bert_config = config['topic_model']
openai_config = config['representation_model']['openai_model']

# Define BERTopic components
def load_topic_modeling(documents):
    logger.info("Loading topic model")
    # Step 1 - Extract embeddings
    embedding_model = SentenceTransformer(embedding_config)
    embeddings = embedding_model.encode(documents, 
                                        show_progress_bar=True)

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(**umap_config)

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(**hdbscan_config)

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(**vectorizer_config, 
                                       preprocessor=preprocess_text)

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()

    # Step 6 - Fine tune representation
    # OpenAI
    client = openai.OpenAI(api_key=openai_api_key)
    prompt_openai = """
    I have a topic that contains the following documents: 
    [DOCUMENTS]
    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Be as specific as possible, but don't use names of entities or countries. Use "GenAI" instead of "Generative AI" or "Genai". Capitalize the first letter of every word. Make sure it is in the following format:
    topic: <topic label>
    """
    openai_model = OpenAI(client,  
                          model='gpt-3.5-turbo',
                          exponential_backoff=True,
                          chat=True,
                          prompt=prompt_openai)
    # KeyBERT
    keybert_model = KeyBERTInspired()

    # MMR to diversify topic representation
    mmr_model = MaximalMarginalRelevance(**mmr_config)

    # All representation models
    representation_model = {
        "MMR": mmr_model,
        "OpenAI": openai_model
    }
    # Run the model
    topic_model = BERTopic(
        embedding_model=embedding_model,          
        umap_model=umap_model,                    
        hdbscan_model=hdbscan_model,              
        vectorizer_model=vectorizer_model,        
        ctfidf_model=ctfidf_model,                
        representation_model=representation_model,         
        **bert_config
    )

    return topic_model, embeddings