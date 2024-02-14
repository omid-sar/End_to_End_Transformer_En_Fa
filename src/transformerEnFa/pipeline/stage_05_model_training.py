
import os
import torch
from pathlib import Path

from transformerEnFa.logging import logger
from transformerEnFa.config.configuration import ConfigurationManager
from transformerEnFa.components.model_training import train_model
from transformerEnFa.utils.model_utils import get_device


class ModelTrainingPipeline():
    def __init__(self, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, model) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.model = model
        self.device = get_device()

        self.config = ConfigurationManager()
        self.config = self.config.get_model_training_config()

    def main(self):
        train_model(
            config=self.config,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            tokenizer_src=self.tokenizer_src,
            tokenizer_tgt=self.tokenizer_tgt,
            model=self.model,
            device=self.device
            )
    



