from transformerEnFa.logging import logger 
from transformerEnFa.config.configuration import ConfigurationManager
from transformerEnFa.utils.model_utils import get_device, latest_weights_file_path
from transformerEnFa.components.model_evaluation import run_validation
from transformerEnFa.components.data_transformation import get_ds
from transformerEnFa.models.transformer import built_transformer


class ModelInferencePipeline:
    def __init__(self):

        self.config_manager = ConfigurationManager()
        self.data_transformation_config = self.config_manager.get_data_transformation_config()
        self.model_config = self.config_manager.get_model_config()
        self.model_training_config = self.config_manager.get_model_training_config()
       
    def main(self):
        pass


       





