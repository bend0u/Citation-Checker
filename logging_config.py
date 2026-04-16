"""
Sci-Verify Logging Configuration
==================================
Sets up dual-output logging:
  - File handler  → sci_verify.log (append mode, full detail, INFO level)
  - Console handler → stderr (WARNING+ only, to avoid cluttering Streamlit output)

All modules use `logging.getLogger(__name__)` which inherits from the root logger
configured here. The log file captures the complete pipeline trace for debugging.
"""

import logging
import os
from datetime import datetime

# Log file lives next to this module (in the project root)
LOG_FILE = os.path.join(os.path.dirname(__file__), "sci_verify.log")


def setup_logging(level=logging.INFO):
    """Configure root logger to write to both file and console.
    
    Called once at app startup (in app.py). Safe to call multiple times —
    guards against duplicate handlers on Streamlit hot-reload.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — append mode, UTF-8 encoding for special characters (emojis, etc.)
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Console handler — only WARNING+ to keep Streamlit terminal clean
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on Streamlit hot-reload (re-import protection)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(LOG_FILE) for h in root.handlers):
        root.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        root.addHandler(console_handler)

    # Write a visible session separator in the log file
    logging.getLogger("sciverify").info(
        f"{'='*60} NEW SESSION {'='*60}"
    )

    return logging.getLogger("sciverify")
