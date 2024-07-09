import os
import logging
import yaml
import pyLDAvis

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")

def setup_logging(log_dir, log_filename='app.log'):
    log_path = os.path.join(log_dir, log_filename)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger(__name__)

    # Save the configuration file
    config_path = 'config.yaml'
    import shutil
    
    destination_path = os.path.join(log_dir, 'config.yaml')
    shutil.copy2(config_path, destination_path)

    return logger

def create_topic_modeling_viz(topic_model, probs, docs):
    import numpy as np
    topic_term_dists = topic_model.c_tf_idf_.toarray()
    outlier = np.array(1 - probs.sum(axis=1)).reshape(-1, 1)
    doc_topic_dists = np.hstack((probs, outlier))

    doc_lengths = [len(doc) for doc in docs]
    vocab = [word for word in topic_model.vectorizer_model.vocabulary_.keys()]
    term_frequency = [topic_model.vectorizer_model.vocabulary_[word] for word in vocab]

    data = {'topic_term_dists': topic_term_dists,
            'doc_topic_dists': doc_topic_dists,
            'doc_lengths': doc_lengths,
            'vocab': vocab,
            'term_frequency': term_frequency}

    viz= pyLDAvis.prepare(**data,n_jobs = 1, mds='mmds')
    
    return viz