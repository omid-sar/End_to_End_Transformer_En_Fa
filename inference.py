from transformerEnFa.logging import logger 
from transformerEnFa.pipeline.stage_07_model_inference import ModelInferencePipeline

sentence = "Who are you?"
sentence = 10

STAGE_NAME = "Model Inference stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    translator = ModelInferencePipeline()
    translator.main(sentence)
    logger.info(f" >>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 