import os
from dotenv import load_dotenv


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")


# Directory
docs_type = "all-docs"  # 'all-docs', 'education, 'others', 'business', 'government'
data_dir = "data/docs"
config_bertopic_dir = "modules/config/config_bertopic.yaml"
bertopic_log_dir = "logs/bertopic/model_65-" + f"{docs_type}"
create_directory("logs/bertopic/model_65" + f"{docs_type}")
# bertopic_log_dir = os.path.join(bertopic_log_dir, bertopic_model_dir)

chatbot_dir = "data/docs/Education and Academia"
config_chatbot_dir = "modules/config/config_chatbot.yaml"
chatbot_log_dir = "logs/chatbot/model_1"
create_directory("logs/chatbot/model_1")
# chatbot_log_dir = os.path(chatbot_log_dir)

# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llama_api_key = os.getenv("LLAMA_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
