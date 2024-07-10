import logging
import os
from constants import *
from helpers import *
from data_loader import *
from text_preprocessing import *
from topic_modeling import *
import json
import pandas as pd
import pyLDAvis

def main():
    logger = setup_logging(log_dir, log_filename='app.log')
    logger.info("Script started")


    # Read and transform data
    if os.path.exists("documents.json"):
        logger.info("Loading documents")
        with open("documents.json", "r") as file:
            all_docs = json.load(file)
    else:
        all_docs = data_loader_subfolders(data_dir)
    
    # Preprocess text and chunk documents
    docs = preprocess_text(all_docs)
    
    # Topic modeling
    try:
        logger.info("Loading topic model")
        topic_model, embeddings = load_topic_modeling(docs)
    except Exception as e:
        logging.error(f"Error loading topic model: {e}")
        raise

    try:
        topics, probs = topic_model.fit_transform(docs)
        logger.info("Topic modeling completed")
    except Exception as e:
        logging.error(f"Error fitting topic modeling: {e}")
        raise

    # Get all documents for all topics
    """docs_by_topic = {}
    topic_info = topic_model.get_topic_info()
    for topic in topic_info.Topic.to_list()[1:]:  # Skip the -1 topic
        docs_by_topic[topic] = topic_model.get_representative_docs(topic) 

    logger.info("Saving topics")
    with open(f'{log_dir}/docs_by_topic.json', 'w') as file:
        json.dump(docs_by_topic, file, indent=2)"""
    
    # Get document info
    doc_info = topic_model.get_document_info(docs)
    doc_info = doc_info[['Document', 'Topic', 'Top_n_words']]
    logger.info("Saving document info")
    doc_info.to_csv(f'{log_dir}/doc_info.csv', index=False)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    logger.info("Saving topic info")
    topic_info.to_csv(f'{log_dir}/topic_info.csv', index=False)

    # Save pyLDAvis visualization
    viz = create_topic_modeling_viz(topic_model, probs, docs)
    logger.info("Saving pyLDAvis visualization")
    pyLDAvis.save_html(viz, f'{log_dir}/pyLDAvis.html')

    logger.info("Script completed")

if __name__ == '__main__':
    main()