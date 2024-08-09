import os
from dotenv import load_dotenv


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")


# Directory
chatbot_dir = "data/Education and Academia"
data_dir = "data/"
config_dir = "modules/config/config.yaml"
docs_type = "all-docs"  # 'all-docs', 'education, 'others', 'business', 'government'

bertopic_log_dir = "logs/bertopic/"
bertopic_model_dir = f"model_65-{docs_type}"
create_directory(bertopic_log_dir + bertopic_log_dir)
bertopic_log_dir = os.path.join(bertopic_log_dir, bertopic_model_dir)

chatbot_log_dir = "logs/chatbot/"
chatbot_model_dir = f"model_"
create_directory(bertopic_log_dir + chatbot_model_dir)
chatbot_log_dir = os.path.join(chatbot_log_dir, chatbot_model_dir)


# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
