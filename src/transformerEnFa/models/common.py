
from pathlib import Path
from transformerEnFa.logging import logger


def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    weights_file_path = str(Path('.') / model_folder / model_filename)
    logger.info(f"Generated weights file path: {weights_file_path}")
    return weights_file_path


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        logger.info(f"No weights files found in {model_folder}. Starting from scratch")
        return None
    weights_files.sort()
    latest_file = str(weights_files[-1])
    logger.info(f"Latest weights file found: {latest_file}")
    return latest_file

