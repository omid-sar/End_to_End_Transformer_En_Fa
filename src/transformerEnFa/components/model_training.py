import os
import torch
import torch.nn as nn
from pathlib import Path

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformerEnFa.logging import logger
from transformerEnFa.utils.common import create_directories
from transformerEnFa.utils.model_utils import get_weights_file_path, latest_weights_file_path
from transformerEnFa.components.model_evaluation import run_validation


def train_model(config, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, model, device):


    create_directories([config.model_folder])
    model = model.to(device)

    writer = SummaryWriter(config.tensorboard_log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)


    initial_epoch = 0
    global_step = 0
    preload =config.preload

    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        logger.info(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        logger.info(f"No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch, config.num_epochs):
            torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f" Processing epoch {epoch:02d}")
            for batch in batch_iterator:
                model.train()

                encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            
                encoder_output = model.encode(encoder_input, encoder_mask) #(B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, decoder_input, decoder_mask, encoder_mask) #(B, seq_len, d_model)
                proj_output = model.project(decoder_output) #(B, seq_len, tgt_vocab_size)

                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)

                # Compute the loss using a simple cross entropy
                #proj_output: (B, seq_len, tgt_vocab_size) -> (B *seq_len, tgt_vocab_size)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # Log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        
                global_step += 1

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.max_len , device, lambda msg: batch_iterator.write(msg), global_step, writer )
            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            model_folder = Path(config.model_folder)
            
            if not Path.exists(model_folder):
                os.mkdir(model_folder)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
            logger.info(f" Model saved in file path : {model_filename}")

        