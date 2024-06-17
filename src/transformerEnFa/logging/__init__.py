import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

absolute_path = os.path.abspath(os.path.dirname(__file__))
base_dir = os.path.abspath(os.path.join(absolute_path, "../../.."))
log_dir = os.path.join(base_dir, "logs")
log_filepath = os.path.join(log_dir,"running_logs.log")

os.makedirs(log_dir, exist_ok=True)




logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("transformerEnFaLogger")

