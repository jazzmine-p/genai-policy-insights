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
import warnings
warnings.filterwarnings('ignore')

def main():
    logger = setup_logging(log_dir, log_filename='app.log')
    logger.info("Script started")


    # Read and transform data
    if os.path.exists(f"data/documents-{docs_type}.json"):
        logger.info("Loading documents")
        with open(f"data/documents-{docs_type}.json", "r") as file:
            docs = json.load(file)
    else:
        docs = data_loader_subfolders(data_dir)
    
    
    # Save preprocessed text for investigation
    processed_docs = []
    for doc in docs:
        processed_doc = preprocess_text(doc)
        processed_docs.append(processed_doc)
    if not os.path.exists(f"data/documents-{docs_type}-preprocessed.json"):
        with open(f"data/documents-{docs_type}-preprocessed.json", "w") as file:
            json.dump(processed_docs, file)

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
    
    # Get document info
    doc_info = topic_model.get_document_info(docs)
    doc_info = doc_info[['Document', 'Topic', 'Top_n_words']]
    logger.info("Saving document info")
    doc_info.to_csv(f'{log_dir}/doc_info.csv', index=False)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    logger.info("Saving topic info")
    topic_info.to_csv(f'{log_dir}/topic_info.csv', index=False)

    # Get topic-term matrix and save
    tfidf = topic_model.c_tf_idf_.toarray()
    vocab = [word for word in topic_model.vectorizer_model.vocabulary_.keys()]
    topic_term_matrix = pd.DataFrame(tfidf, columns=vocab)
    logger.info("Saving topic-term matrix")
    topic_term_matrix.to_csv(f'{log_dir}/topic_term_matrix.csv')
    # Save vocab
    with open(f'{log_dir}/vocab.json', 'w') as file:
        json.dump(vocab, file)

    # PyLDAvis visualization
    viz = create_topic_modeling_viz(topic_model, probs, docs)
    logger.info("Saving pyLDAvis visualization")
    pyLDAvis.save_html(viz, f'{log_dir}/pyLDAvis.html')
    pyLDAvis.save_json(viz, f'{log_dir}/pyLDAvis.json')

    viz_processed = create_topic_modeling_viz(topic_model, probs, processed_docs)
    pyLDAvis.save_html(viz_processed, f'{log_dir}/pyLDAvis-processed.html')

    logger.info("Script completed")

if __name__ == '__main__':
    main()