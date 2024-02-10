import os
from pathlib import Path
from datasets import load_dataset

from transformerEnFa.logging import logger
from transformerEnFa.utils.common import get_size
from transformerEnFa.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
                   
    def download_file(self):
        dataset_path = Path(self.config.local_data_file)
        if not dataset_path.exists():
            dataset = load_dataset(self.config.dataset_name, split='train[:5%]')
            dataset.save_to_disk(dataset_path)  
            logger.info(f"{self.config.dataset_name} downloaded and saved to {dataset_path}!")
        else:
            logger.info(f"Dataset already exists at {dataset_path}. Size: {get_size(dataset_path)}")
