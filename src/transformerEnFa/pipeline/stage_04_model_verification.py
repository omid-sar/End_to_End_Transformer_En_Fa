import torch
from pathlib import Path
from transformerEnFa.utils.common import  create_directories
from transformerEnFa.utils.model_utils import get_device
from transformerEnFa.models.transformer import built_transformer
from transformerEnFa.config.configuration import ConfigurationManager
from transformerEnFa.utils.model_utils import save_model_summary, save_initial_weights
from transformerEnFa.logging import logger

class ModelVerificationTrainingPipeline:
    def __init__(self, tokenizer_src, tokenizer_tgt):

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_vocab_size = tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_model_config()
        self.device = get_device()
       


    def main(self):
        try:
            # Instantiate the model to check for syntax errors in initialization
            model = built_transformer(
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_seq_len=self.config.src_seq_len,
                tgt_seq_len=self.config.tgt_seq_len,
                d_model=self.config.d_model,
                N=self.config.N,
                h=self.config.h,
                dropout=self.config.dropout,
                d_ff=self.config.d_ff
            ).to(self.device)
            logger.info("Model instantiation successful.")
            create_directories([self.config.verification_info_dir])
            
            # Optionally, perform a simple forward pass check
            # dummy_input = torch.rand(1, self.config.src_seq_len).long().to(self.device)
            # with torch.no_grad():
            #     _ = model(dummy_input)
            # logger.info("Basic forward pass successful.")

        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            raise e

        # Save model summary and initial weights as before
        save_model_summary(
            model,
            Path(self.config.verification_info_dir) / self.config.verification_summary_file,
            input_size=(self.config.src_seq_len,),
            device=str(self.device)
        )
        logger.info("Model and device setup complete.")
        return model
    
