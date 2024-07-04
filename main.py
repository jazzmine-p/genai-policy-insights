import logging
import os
from dotenv import load_dotenv
from data_loader import convert_pdfs_to_markdown, split_markdown_by_section, save_sections_to_list, data_loader_subfolders
from text_preprocessing import filter_sections, preprocess_text
from topic_modeling import load_topic_modeling

# Set up logger
def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("app.log"),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger(__name__)
    return logger

# Main
def main():
    logger = setup_logging()
    logger.info("Script started")

    # Load API keys
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    unstructured_api_key = os.getenv('UNSTRUCTURED_API_KEY')

    # Read and transform data
    main_directory = "data/"
    sections_list = data_loader_subfolders(main_directory)

    # Preprocess text and chunk documents
    filtered_sections = filter_sections(sections_list)
    filtered_sections = preprocess_text(filtered_sections)

    # Topic modeling
    topic_model, embeddings = load_topic_modeling(filtered_sections)
    logger.info("Training topic model")
    topics, probs = topic_model.fit_transform(filtered_sections, embeddings)
    logger.info("Topic modeling completed")

    # Show topics
    logger.info("Retrieving topic info")
    topic_info = topic_model.get_topic_info()
    logger.info(f"Topic Info: {topic_info}")

    logger.info("Script completed")

if __name__ == '__main__':
    main()