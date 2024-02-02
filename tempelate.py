import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

project_name = ""
list_of_files = [
    "github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/data_utils.py",  # For data-related utility functions
    f"src/{project_name}/utils/model_utils.py",  # For model-related utility functions
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/data_ingestion.py",  # For data ingestion processes
    f"src/{project_name}/pipeline/data_preprocessing.py",  # For data preprocessing steps
    f"src/{project_name}/pipeline/model_training.py",  # For model training processes
    f"src/{project_name}/pipeline/model_evaluation.py",  # For model evaluation processes
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/models/__init__.py",  # For model definitions
    f"src/{project_name}/constants/__init__.py",
    "config/config.yml",
    "params.yml",
    "app.py",  # For model serving/application interface
    "main.py",  # Main script to run the pipeline
    "Dockerfile",  # For containerization
    "artifacts/.gitkeep",  # Placeholder for artifacts directory
]

# Create the files and directories
for filepath in list_of_files:
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Created empty file {filepath}")

    else:
        logging.info(f"File {filepath} already exists")


