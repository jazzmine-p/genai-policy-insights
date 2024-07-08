import os
import logging
import yaml

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def setup_logging(log_dir):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_dir),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger(__name__)

    # Save the configuration file
    config = '/config.yaml'
    config_path = os.path.join(log_dir, 'config.yaml')
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    return logger