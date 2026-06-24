import os
import logging
import inspect

log_level = os.environ.get("LOG_LEVEL", "CRITICAL").upper()
log_level = getattr(logging, log_level, logging.CRITICAL)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] (%(filename)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def get_logger() -> logging.Logger:
    filename = inspect.stack()[1].filename
    logger_name = os.path.basename(filename)

    return logging.getLogger(logger_name)
