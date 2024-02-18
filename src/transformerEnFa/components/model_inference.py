import os
import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset, load_from_disk
from transformerEnFa.logging import logger
from transformerEnFa.utils.model_utils import get_device, latest_weights_file_path
from transformerEnFa.config.configuration import ConfigurationManager
from transformerEnFa.components.data_transformation import get_or_build_tokenizer
from transformerEnFa.models.transformer import built_transformer
from transformerEnFa.components.data_transformation import BilingualDataset



class Inference:
    def __init__(self, data_transformation_config, model_config, model_training_config):
        self.data_transformation_config = data_transformation_config
        self.model_config = model_config
        self.model_training_config = model_training_config
        self.device = get_device()
        
        self.tokenizer_src = get_or_build_tokenizer(config=self.data_transformation_config, ds=None, lang=self.data_transformation_config.lang_src)
        self.tokenizer_tgt = get_or_build_tokenizer(config=self.data_transformation_config, ds=None, lang=self.data_transformation_config.lang_tgt)
        
        src_vocab_size = self.tokenizer_src.get_vocab_size()
        tgt_vocab_size = self.tokenizer_tgt.get_vocab_size()

        self.model = built_transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_seq_len=self.model_config.src_seq_len,
            tgt_seq_len=self.model_config.tgt_seq_len,
            d_model=self.model_config.d_model,
            N=self.model_config.N,
            h=self.model_config.h,
            dropout=self.model_config.dropout,
            d_ff=self.model_config.d_ff
        ).to(self.device)

        # Load the pretrained weights
        model_filename = latest_weights_file_path(self.model_training_config)
        state = torch.load(model_filename)
        self.model.load_state_dict(state['model_state_dict'])

    def translate(self, sentence: str):
        label = ""
        if isinstance(sentence, int) or sentence.isdigit():
            id = int(sentence)
            dataset_path = Path(self.data_transformation_config.local_data_file)
            ds_raw = load_from_disk(dataset_path)
            ds = BilingualDataset(
                ds_raw,
                self.tokenizer_src, 
                self.tokenizer_tgt, 
                self.data_transformation_config.lang_src, 
                self.data_transformation_config.lang_tgt, 
                self.data_transformation_config.seq_len
            )
            sentence = ds[id]['src_text']
            label = ds[id]['tgt_text']

        seq_len = self.data_transformation_config.seq_len

        self.model.eval()
        with torch.no_grad():
            source_encoded = self.tokenizer_src.encode(sentence)
            source = torch.cat([
                torch.tensor([self.tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
                torch.tensor(source_encoded.ids, dtype=torch.int64),
                torch.tensor([self.tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
                torch.tensor([self.tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source_encoded.ids) - 2), dtype=torch.int64)
            ], dim=0).to(self.device)

            source_mask = (source != self.tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(self.device)
            encoder_output = self.model.encode(source, source_mask)

            decoder_input = torch.empty(1, 1).fill_(self.tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(self.device)

            if label != "":
                print(f"{f'ID: ':>12}{id}") 
            print(f"{f'SOURCE: ':>12}{sentence}")
            if label != "":
                print(f"{f'TARGET: ':>12}{label}") 
            print(f"{f'PREDICTED: ':>12}", end='')

            while decoder_input.size(1) < seq_len:
                decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(self.device)
                out = self.model.decode(encoder_output, decoder_input, decoder_mask, source_mask)

                prob = self.model.project(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(self.device)], dim=1)

                print(f"{self.tokenizer_tgt.decode([next_word.item()])} \n", end=' ')

                if next_word == self.tokenizer_tgt.token_to_id('[EOS]'):
                    break

        return self.tokenizer_tgt.decode(decoder_input[0].tolist(), skip_special_tokens=True)
