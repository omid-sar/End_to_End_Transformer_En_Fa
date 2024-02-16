from transformerEnFa.logging import logger 
from transformerEnFa.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from transformerEnFa.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from transformerEnFa.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from transformerEnFa.pipeline.stage_04_model_verification import ModelVerificationTrainingPipeline
from transformerEnFa.pipeline.stage_05_model_training import ModelTrainingPipeline
from transformerEnFa.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f"\n\nx{'=' * 80}x \n\n>>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 


STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = data_transformation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 


STAGE_NAME = "Model Validation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_verification_pipeline = ModelVerificationTrainingPipeline(tokenizer_src, tokenizer_tgt)
    model = model_verification_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "Model Training stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_training = ModelTrainingPipeline(train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, model)
    model_training.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "Model Evaluation stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_training = ModelEvaluationPipeline()
    model_training.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 