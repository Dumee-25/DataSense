"""Centralised logging configuration for DataSense.

Sets up:
  - Console handler  — coloured, concise output for development.
  - Rotating file handler — structured logs written to ``logs/datasense.log``
    with automatic rotation (5 MB per file, 5 backups kept).

Usage (called once at app startup):
    from utils.logging_config import setup_logging
    setup_logging()

Every module that already does ``logger = logging.getLogger(__name__)``
will automatically pick up these handlers — no changes needed elsewhere.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# ── Configurable via environment ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "datasense.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# ── Formats ───────────────────────────────────────────────────────────────────
CONSOLE_FMT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
FILE_FMT = "%(asctime)s [%(levelname)-8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure the root logger with console + rotating file handlers."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, LOG_FILE)

    root = logging.getLogger()

    # Avoid adding duplicate handlers if called more than once (e.g. uvicorn reload)
    if root.handlers:
        return

    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # ── Console handler ───────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(CONSOLE_FMT, datefmt=DATE_FMT))
    root.addHandler(console)

    # ── Rotating file handler ─────────────────────────────────────────────
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FMT, datefmt=DATE_FMT))
    root.addHandler(file_handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised  ·  level=%s  ·  file=%s  ·  max=%s MB  ·  backups=%d",
        LOG_LEVEL, log_path, LOG_MAX_BYTES // (1024 * 1024), LOG_BACKUP_COUNT,
    )
