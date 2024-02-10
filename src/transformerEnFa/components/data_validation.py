import os
from transformerEnFa.logging import logger

from transformerEnFa.entity import DataValidationConfig


class DataValidation:
    def __init__(self, config):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = None

            all_files = os.listdir(os.path.join("artifacts", "data_ingestion", "data"))
            missing_files = []

            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in all_files:
                    validation_status = False
                    missing_files.append(required_file)
                    logger.error(f"Missing required file: {required_file}")
                else:
                    validation_status = True
                    logger.info(f"Found required file: {required_file}")

            if missing_files:
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}\n")
                    f.write(f"Missing files: {', '.join(missing_files)}")
                logger.error(f"Validation failed. Missing files: {', '.join(missing_files)}")
            else:
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
                logger.info("All required files are present.")

            return validation_status

        except Exception as e:
            logger.exception("Failed during file validation.")
            raise e
