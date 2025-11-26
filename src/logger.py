import logging
import sys
from loguru import logger
from enum import StrEnum

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        logger.opt(depth=6, exception=record.exc_info).log(
            level, record.getMessage()
        )


class LogLevel(StrEnum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def setup_logging(
    app_level: LogLevel = LogLevel.DEBUG,
    third_party_level: LogLevel = LogLevel.INFO,
):
    # -------------------------------
    # 1. Configure Loguru (your code)
    # -------------------------------
    logger.remove()  # remove default sink
    logger.add(sys.stderr, level=app_level)

    # -----------------------------------------------
    # 2. Configure standard logging (third-party logs)
    # -----------------------------------------------
    # Clean root handlers
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    # Root logger receives only WARNING+ (or your choice)
    logging.root.setLevel(third_party_level)

    # Every standard logger uses InterceptHandler
    logging.basicConfig(handlers=[InterceptHandler()], level=third_party_level)

    # OPTIONAL: set level for very noisy libraries:
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

