import os
from dotenv import load_dotenv

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")

# Directory
chatbot_dir = 'data/'
data_dir = 'data/'
save_dir = 'results/'
docs_type = 'all-docs' # 'all-docs', 'education, 'others'
model_dir = f'model_65-{docs_type}'
create_directory(save_dir + model_dir)
log_dir = os.path.join(save_dir, model_dir)

# Load API keys
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
llama_api_key = os.getenv("LLAMA_API_KEY")