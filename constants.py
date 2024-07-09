import os
from dotenv import load_dotenv
from helpers import create_directory

# Load the environment variables from the .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
unstructured_api_key = os.getenv('UNSTRUCTURED_API_KEY')

# Directory
data_dir = 'data/'
save_dir = 'results/'
model_dir = 'model2'
create_directory(save_dir + model_dir)
log_dir = os.path.join(save_dir, model_dir)