import os
from dotenv import load_dotenv
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
import yaml
import openai
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
import logging

logger = logging.getLogger(__name__)


# Load the YAML configuration file
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Load API keys
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Access hyperparameters for each model
umap_config = config['umap_model']
hdbscan_config = config['hdbscan_model']
vectorizer_config = config['vectorizer_model']
mmr_config = config['mmr_model']
bert_config = config['topic_model']

# Define BERTopic components
def load_topic_modeling(documents):
    logger.info("Loading topic model")
    # Step 1 - Extract embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(documents, show_progress_bar=True)

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(**umap_config)

    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(**hdbscan_config)

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(**vectorizer_config, ngram_range=(1, 3))

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()

    # GPT-3.5
    client = openai.OpenAI(api_key=openai_api_key)
    prompt = """
    I have a topic that contains the following documents: 
    [DOCUMENTS]
    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
    topic: <topic label>
    """
    openai_model = OpenAI(client, model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)

    # All representation models
    representation_model = {
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