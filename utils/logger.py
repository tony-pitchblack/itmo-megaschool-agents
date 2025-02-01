import sys

from aiologger import Logger
from aiologger.formatters.base import Formatter
from aiologger.handlers.files import AsyncFileHandler
from aiologger.handlers.streams import AsyncStreamHandler
from aiologger.levels import LogLevel

async def flush_aiologger(logger: Logger):
    """Flush all handlers of an async logger."""
    for handler in logger.handlers:
        await handler.flush()

async def setup_logger(log_level=LogLevel.INFO):
    # Create logger instance
    logger = Logger(name="api_logger")

    # Create formatter
    formatter = Formatter(
        fmt="{asctime} | {levelname} | {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    file_handler = AsyncFileHandler(
        filename="logs/api.log",
        mode="a",
        encoding="utf-8",
    )
    stream_handler = AsyncStreamHandler(stream=sys.stdout)
    file_handler.formatter = formatter
    stream_handler.formatter = formatter
    logger.add_handler(file_handler)
    logger.add_handler(stream_handler)

    logger.level = log_level
    print(f"Set up a logger with log level: {log_level}")

    return logger
