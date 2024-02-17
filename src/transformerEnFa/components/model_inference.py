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


config_manager = ConfigurationManager()
data_transformation_config = config_manager.get_data_transformation_config()
model_config = config_manager.get_model_config()
model_training_config = config_manager.get_model_training_config()

sentence = "What does mean?"
sentence = 1 

device = get_device()
tokenizer_src = get_or_build_tokenizer(config=data_transformation_config, ds=None, lang=data_transformation_config.lang_src)
tokenizer_tgt = get_or_build_tokenizer(config=data_transformation_config, ds=None, lang=data_transformation_config.lang_tgt)
src_vocab_size = tokenizer_src.get_vocab_size()
tgt_vocab_size= tokenizer_tgt.get_vocab_size()

model = built_transformer( 
    src_vocab_size = src_vocab_size ,
    tgt_vocab_size= tgt_vocab_size,
    src_seq_len = model_config.src_seq_len,
    tgt_seq_len = model_config.tgt_seq_len,
    d_model = model_config.d_model,
    N = model_config.N,
    h = model_config.h,
    dropout = model_config.dropout,
    d_ff = model_config.d_ff
).to(device)


# Load the pretrained weights
model_filename = latest_weights_file_path(model_training_config)
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])


# if the sentence is a number use it as an index to the test set
label = ""
if type(sentence) == int or sentence.isdigit():
    id = int(sentence)
    dataset_path = Path(data_transformation_config.local_data_file)
    ds_raw = load_from_disk(dataset_path)
    ds = BilingualDataset(
        ds_raw,
        tokenizer_src, 
        tokenizer_tgt, 
        data_transformation_config.lang_src, 
        data_transformation_config.lang_tgt, 
        data_transformation_config.seq_len
    )
    sentence = ds[id]['src_text']
    label = ds[id]['tgt_text']
seq_len = data_transformation_config.seq_len

model.eval()
with torch.no_grad():
# Precompute the encoder output and reuse it for every generation step
    source = tokenizer_src.encode(sentence)
    source = torch.cat([
    torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
    torch.tensor(source.ids, dtype=torch.int64),
    torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
    torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
], dim=0).to(device)
source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
encoder_output = model.encode(source, source_mask)

# Initialize the decoder input with the sos token
decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

# Print the source sentence and target start prompt
if label != "": print(f"{f'ID: ':>12}{id}") 
print(f"{f'SOURCE: ':>12}{sentence}")
if label != "": print(f"{f'TARGET: ':>12}{label}") 
print(f"{f'PREDICTED: ':>12}", end='')

# Generate the translation word by word
while decoder_input.size(1) < seq_len:
    # build mask for target and calculate output
decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

# project next token
prob = model.project(out[:, -1])
_, next_word = torch.max(prob, dim=1)
decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

# print the translated word
print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

# break if we predict the end of sentence token
if next_word == tokenizer_tgt.token_to_id('[EOS]'):
    break

# convert ids to tokens
return tokenizer_tgt.decode(decoder_input[0].tolist())

#read sentence from argument
translate(sys.argv[1] if len(sys.argv) > 1 else "I am not a very good a student.")