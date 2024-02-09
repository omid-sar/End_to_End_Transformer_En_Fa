
import os
from datasets import load_dataset 
from torch.utils.data import DataLoader,  random_split

from transformerEnFa.logging import logger
from transformerEnFa.config.configuration import ConfigurationManager
from transformerEnFa.components.data_transformation import BilingualDataset, get_or_build_tokenizer, get_all_sentences


class DataTransformationTrainingPipeline():
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_data_transformation_config()

    def calculate_max_lengths(self, ds_raw):
        max_len_src = 0
        max_len_tgt = 0
        for item in ds_raw:
            src_ids = self.tokenizer_src.encode(item['translation'][self.config.lang_src]).ids
            tgt_ids = self.tokenizer_tgt.encode(item['translation'][self.config.lang_tgt]).ids
            max_len_src = max(len(src_ids), max_len_src)
            max_len_tgt = max(len(tgt_ids), max_len_tgt)
        return max_len_src, max_len_tgt

    def get_ds(self):
        # Load the dataset
        ds_raw = load_dataset(self.config.dataset_name, split='train[:5%]')

        # Build tokenizers for source and target languages
        self.tokenizer_src = get_or_build_tokenizer(self.config, ds_raw, self.config.lang_src)
        self.tokenizer_tgt = get_or_build_tokenizer(self.config, ds_raw, self.config.lang_tgt)

        # Split the dataset into training and validation sets
        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

        # Prepare the datasets for training and validation
        train_ds = BilingualDataset(train_ds_raw, self.tokenizer_src, self.tokenizer_tgt, self.config.lang_src, self.config.lang_tgt, self.config.seq_len)
        val_ds = BilingualDataset(val_ds_raw, self.tokenizer_src, self.tokenizer_tgt, self.config.lang_src, self.config.lang_tgt, self.config.seq_len)

        # Calculate the maximum sentence lengths
        max_len_src, max_len_tgt = self.calculate_max_lengths(ds_raw)

        # Log the maximum sentence lengths
        logger.info(f"Max length of source sentence: {max_len_src}")
        logger.info(f"Max length of target sentence: {max_len_tgt}")

        # Create DataLoaders for the training and validation datasets
        self.train_dataloader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

        return self.train_dataloader, self.val_dataloader, self.tokenizer_src, self.tokenizer_tgt
    
