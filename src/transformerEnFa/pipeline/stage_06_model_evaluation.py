from transformerEnFa.logging import logger 
from transformerEnFa.config.configuration import ConfigurationManager
from transformerEnFa.utils.model_utils import get_device, latest_weights_file_path
from transformerEnFa.components.model_evaluation import run_validation
from transformerEnFa.components.data_transformation import get_ds
from transformerEnFa.models.transformer import built_transformer


class ModelEvaluationPipeline:
    def __init__(self, tokenizer_src, tokenizer_tgt):

        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_vocab_size = tokenizer_src.get_vocab_size()
        self.tgt_vocab_size = tokenizer_tgt.get_vocab_size()

        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_model_evaluation_config()
        self.data_tranformation_config = self.config_manager.get_data_transformation_config()
        self.model_config = self.config_manager.get_model_config()
        self.model_training_config = self.config_manager.get_model_training_config()
        self.device = get_device()
       
    def main(self):
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(self.data_tranformation_config)

        model = built_transformer(
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                src_seq_len=self.model_config.src_seq_len,
                tgt_seq_len=self.model_config.tgt_seq_len,
                d_model=self.model_config.d_model,
                N=self.model_config.N,
                h=self.model_config.h,
                dropout=self.model_config.dropout,
                d_ff=self.model_config.d_ff
            ).to(self.device)
        
     
        model_filename = latest_weights_file_path(self.model_training_config)
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])

        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=10)    

       





