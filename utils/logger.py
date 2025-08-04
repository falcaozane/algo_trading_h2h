# utils/logger.py

import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    Sets up a logger that writes to both console and a timestamped log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"log_{timestamp}.log")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logger initialized.")

