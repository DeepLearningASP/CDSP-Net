"""Logging utilities."""

import logging
import os
from typing import Optional


_LOG_FORMAT = "[ %(asctime)s ] %(message)s"
_DATE_FORMAT = "%a %b %d %H:%M:%S %Y"


def _create_logger(
    name: str,
    log_file_path: Optional[str],
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    if log_file_path is not None:
        os.makedirs(
            os.path.dirname(log_file_path),
            exist_ok=True,
        )
        file_handler = logging.FileHandler(
            log_file_path,
            mode="w",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt=_LOG_FORMAT,
                datefmt=_DATE_FORMAT,
            )
        )
        logger.addHandler(file_handler)

    return logger


def load_logger(
    log_path: str,
    if_save: bool = True,
) -> logging.Logger:
    log_file_path = os.path.join(log_path, "log.txt") if if_save else None
    logger_name = f"train:{os.path.abspath(log_path)}"
    return _create_logger(
        logger_name,
        log_file_path,
    )


def load_test_logger(
    test_log_path: str,
    if_save: bool = True,
) -> logging.Logger:
    log_file_path = (
        os.path.join(
            test_log_path,
            "test_results.txt",
        )
        if if_save
        else None
    )
    logger_name = f"test:{os.path.abspath(test_log_path)}"
    return _create_logger(
        logger_name,
        log_file_path,
    )
