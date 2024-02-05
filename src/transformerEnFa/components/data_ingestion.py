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
        if not os.path.exists(self.config.local_data_file):
            dataset = load_dataset(self.config.dataset_name)
            os.makedirs(self.config.local_data_file, exist_ok=True)
            # since the dataset only has train
            dataset['train'].to_csv(os.path.join(self.config.local_data_file, "ds_raw.csv"))
            logger.info(f"{self.config.dataset_name} download!")
        else:
            file_path = Path(self.config.local_data_file) / 'ds_raw.csv'
            logger.info(f"File already exists of size: {get_size(file_path)}")
           
