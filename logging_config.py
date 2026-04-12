"""
Sci-Verify Logging Configuration
Writes structured logs to sci_verify.log with timestamps,
module names, and log levels.
"""

import logging
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "sci_verify.log")


def setup_logging(level=logging.INFO):
    """Configure root logger to write to both file and console."""
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — append mode, UTF-8
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler — less verbose
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on re-import
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(LOG_FILE) for h in root.handlers):
        root.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(console_handler)

    # Write session separator
    logging.getLogger("sciverify").info(
        f"{'='*60} NEW SESSION {'='*60}"
    )

    return logging.getLogger("sciverify")
