"""Logging configuration for MedQuery.

Provides structured logging with verbose mode support for both
CLI and programmatic usage.

Usage:
    # CLI mode (configured via --verbose flag)
    from medquery.logging import setup_logging
    setup_logging(verbose=True)

    # Module-level logging
    from medquery.logging import get_logger
    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
"""

import logging
import sys
from typing import Literal

# Package-wide logger
PACKAGE_NAME = "medquery"

# Log format templates
FORMATS = {
    "simple": "%(levelname)s: %(message)s",
    "detailed": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "debug": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
}

# Color codes for terminal output
COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """Formatter with color support for terminal output."""

    def __init__(self, fmt: str, use_colors: bool = True):
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = COLORS.get(record.levelname, "")
            reset = COLORS["RESET"]
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


class LoggerState:
    """Global logger state management."""

    _configured: bool = False
    _level: int = logging.WARNING
    _verbose: bool = False

    @classmethod
    def is_configured(cls) -> bool:
        return cls._configured

    @classmethod
    def set_configured(cls, level: int, verbose: bool) -> None:
        cls._configured = True
        cls._level = level
        cls._verbose = verbose

    @classmethod
    def get_level(cls) -> int:
        return cls._level

    @classmethod
    def is_verbose(cls) -> bool:
        return cls._verbose


def setup_logging(
    verbose: bool = False,
    debug: bool = False,
    quiet: bool = False,
    log_file: str | None = None,
    format_style: Literal["simple", "detailed", "debug"] = "simple",
) -> logging.Logger:
    """Configure logging for the entire package.

    Args:
        verbose: Enable verbose output (INFO level)
        debug: Enable debug output (DEBUG level, overrides verbose)
        quiet: Suppress all output except errors (ERROR level)
        log_file: Optional file path for logging
        format_style: Log format style

    Returns:
        Root package logger
    """
    # Determine log level
    if debug:
        level = logging.DEBUG
        format_style = "debug"
    elif quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Get package logger
    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(FORMATS[format_style])
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(FORMATS["debug"])
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Mark as configured
    LoggerState.set_configured(level, verbose or debug)

    logger.debug(f"Logging configured: level={logging.getLevelName(level)}, verbose={verbose}, debug={debug}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Processing query")
    """
    # Ensure name is under package namespace
    if not name.startswith(PACKAGE_NAME):
        name = f"{PACKAGE_NAME}.{name}"

    logger = logging.getLogger(name)

    # Don't set level on child loggers - let them inherit from parent
    # This ensures setup_logging() affects all loggers in the package

    return logger


def set_level(level: int | str) -> None:
    """Set logging level for the package.

    Args:
        level: Logging level (e.g., logging.DEBUG, "DEBUG")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = logging.getLogger(PACKAGE_NAME)
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)


def enable_verbose() -> None:
    """Enable verbose logging (INFO level)."""
    set_level(logging.INFO)


def enable_debug() -> None:
    """Enable debug logging (DEBUG level)."""
    set_level(logging.DEBUG)


def disable_logging() -> None:
    """Disable all logging output."""
    logging.getLogger(PACKAGE_NAME).handlers.clear()


# Convenience function for quick verbose enable
def verbose(enabled: bool = True) -> None:
    """Quick toggle for verbose mode.

    Args:
        enabled: Whether to enable verbose mode
    """
    if enabled:
        setup_logging(verbose=True)
    else:
        setup_logging(quiet=True)
