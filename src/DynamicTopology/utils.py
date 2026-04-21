import logging


def log_debug(logger: logging.Logger, msg: str):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(msg)
