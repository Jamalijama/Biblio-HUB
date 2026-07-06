import logging
import traceback
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "biblio_hub.log"


def get_app_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("biblio_hub")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def log_exception(context: str, exc: Exception) -> None:
    logger = get_app_logger()
    logger.error("%s | %s\n%s", context, exc, traceback.format_exc())
