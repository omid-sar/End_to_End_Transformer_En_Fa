
import os
import torch
from pathlib import Path

from transformerEnFa.logging import logger
from transformerEnFa.config.configuration import ConfigurationManager
from transformerEnFa.components.data_transformation import get_ds


class DataTransformationTrainingPipeline():
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_tranformation_config = config.get_data_transformation_config()
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config=data_tranformation_config)

        save_dir = Path(data_tranformation_config.root_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(train_dataloader, save_dir / 'train_dataloader.pth')
        torch.save(val_dataloader, save_dir / 'val_dataloader.pth')

        logger.info("Data transformation stage completed and outputs saved.")
        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


