import os
import logging
import yaml
import pyLDAvis

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")


def setup_logging(log_dir, log_filename="app.log"):
    log_path = os.path.join(log_dir, log_filename)

    with open(log_path, "w"):
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # Save the configuration file
    config_path = "modules/config/config.yaml"
    import shutil

    destination_path = os.path.join(log_dir, "config.yaml")
    shutil.copy2(config_path, destination_path)

    return logger
