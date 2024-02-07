from transformerEnFa.constants import *
from transformerEnFa.utils.common import read_yaml, create_directories
from transformerEnFa.entity import DataIngestionConfig
from transformerEnFa.entity import DataValidationConfig
from transformerEnFa.entity import ModelConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_name=config.dataset_name,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config
    
    def get_data_validation_config(self)-> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
        root_dir = config.root_dir,
        STATUS_FILE = config.STATUS_FILE,
        ALL_REQUIRED_FILES= config.ALL_REQUIRED_FILES
        )

        return data_validation_config
    
    def get_model_config(self) -> ModelConfig:
        config = self.config.model_config

        create_directories([config.root_dir])
        create_directories([config.verification_info_dir])
        

        return ModelConfig(
            root_dir = config.root_dir,
            verification_info_dir = config.verification_info_dir,
            verification_summary_file = config.verification_summary_file, 
            verification_weights_file = config.verification_weights_file, 
            src_vocab_size = config.src_vocab_size,
            tgt_vocab_size = config.tgt_vocab_size,
            src_seq_len = config.src_seq_len,
            tgt_seq_len = config.tgt_seq_len,
            d_model = config.d_model,
            N = config.N,
            h = config.h,
            dropout = config.dropout,
            d_ff = config.d_ff
        )

