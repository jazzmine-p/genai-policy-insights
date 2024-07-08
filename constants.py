import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
unstructured_api_key = os.getenv('UNSTRUCTURED_API_KEY')

# Data file paths
data_dir = 'data/'
save_dir = 'results/'