from pathlib import Path
import torch

from transformerEnFa.logging import logger


def get_weights_file_path(config, epoch):
    model_folder = config.model_folder
    model_filename = f"{config.model_basename}{epoch}.pt"
    weights_file_path = str(Path('.') / model_folder / model_filename)
    logger.info(f"Generated weights file path: {weights_file_path}")
    return weights_file_path



def latest_weights_file_path(config):
    model_folder = config.model_folder
    model_filename = f"{config.model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if not weights_files:
        logger.info(f"No weights files found in {model_folder}. Starting from scratch")
        return None
    weights_files.sort()
    latest_file = str(weights_files[-1])
    logger.info(f"Latest weights file found: {latest_file}")
    return latest_file


def save_model_summary(model, file_path, input_size, device='cpu'):
    """
    Saves the model summary to a file.
    """
    original_device = next(model.parameters()).device
    model.to(device)

    try:
        with open(file_path, 'w') as f:
            # Here you would generate the model summary.
            # For now, we're just simulating this by writing a placeholder string.
            f.write("Model summary placeholder")
        logger.info(f"Model summary saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save model summary: {e}")
    finally:
        model.to(original_device)

def save_initial_weights(model, file_path):
    """
    Saves the initial weights of the model.
    """
    try:
        torch.save(model.state_dict(), file_path)
        logger.info(f"Model initial weights saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save initial weights: {e}")
