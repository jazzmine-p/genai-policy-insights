import logging
import os
from modules.config.constants import *
from modules.helpers import *
from modules.data_loader import *
from modules.bertopic.text_preprocessing import *
from modules.bertopic.topic_modeling import *
from modules.bertopic.visualization import *
from bertopic import *
import json
import pandas as pd
import plotly.io as pio
import warnings

warnings.filterwarnings("ignore")

def main():
    logger = setup_logging(bertopic_log_dir, log_filename="app.log")
    logger.info(f"Script for {docs_type} started")

    # Read and transform data
    if os.path.exists(f"data/documents-{docs_type}.json"):
        logger.info("Loading documents")
        with open(f"data/documents-{docs_type}.json", "r") as file:
            docs = json.load(file)
    else:
        docs = data_loader_subfolders(data_dir)

    """
    # Save preprocessed text for investigation
    processed_docs = []
    for doc in docs:
        processed_doc = preprocess_text(doc)
        processed_docs.append(processed_doc)
    if not os.path.exists(f"data/documents-{docs_type}-preprocessed.json"):
        with open(f"data/documents-{docs_type}-preprocessed.json", "w") as file:
            json.dump(processed_docs, file)
    """

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
    doc_info = doc_info[["Document", "Topic", "Top_n_words"]]
    logger.info("Saving document info")
    doc_info.to_csv(f"{bertopic_log_dir}/doc_info.csv", index=False)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    logger.info("Saving topic info")
    topic_info.to_csv(f"{bertopic_log_dir}/topic_info.csv", index=False)

    # Get custom topic labels
    topic_label = {}
    topic_info["OpenAI"] = (
        topic_info["OpenAI"].astype(str).str.replace("['", "").str.replace("']", "")
    )
    for i in range(len(topic_info)):
        topic_label[topic_info["Topic"][i]] = topic_info["OpenAI"][i]
    topic_model.set_topic_labels(topic_label)

    # Get topic-term matrix and save
    tfidf = topic_model.c_tf_idf_.toarray()
    vocab = [word for word in topic_model.vectorizer_model.vocabulary_.keys()]
    topic_term_matrix = pd.DataFrame(tfidf, columns=vocab)
    logger.info("Saving topic-term matrix")
    topic_term_matrix.to_csv(f"{bertopic_log_dir}/topic_term_matrix.csv")
    logger.info("Saving vocab")
    with open(f"{bertopic_log_dir}/vocab.json", "w") as file:
        json.dump(vocab, file)

    # BERTopic visualization
    logger.info("Saving BERTopic visualizations")
    intertopic_distance = topic_model.visualize_topics(custom_labels=True)
    pio.write_html(
        intertopic_distance, file=f"{bertopic_log_dir}/intertopic_distance.html", auto_open=False
    )
    datamap = topic_model.visualize_document_datamap(
        docs=docs, embeddings=embeddings, custom_labels=True
    )
    datamap.savefig(f"{bertopic_log_dir}/datamap.png")
    visualize_topic_term(topic_model)
    visualize_topic_hierarchy(topic_model, docs)

    # Save model
    logger.info("Saving topic model")
    embedding_model = config["embedding_model"]["sentence-transformers"]
    topic_model.save(
        f"{bertopic_log_dir}/topic_model",
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=embedding_model,
    )
    logger.info("Script completed")


if __name__ == "__main__":
    main()
