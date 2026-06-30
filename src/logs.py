import logging
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

from src import config

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level, logging.INFO)


def setup_logging(calling_file: str):
    script_name = Path(calling_file).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hostname = socket.gethostname()
    log_filename = f"{script_name}_{timestamp}_{hostname}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] (%(filename)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    log_dir = config.LOGS_DIR_PATH
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. Combine them into an absolute path
    log_file_path = log_dir / log_filename
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 5. Console Handler (prints to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)