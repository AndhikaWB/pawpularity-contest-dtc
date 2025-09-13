import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def enable_logging():
    """Stream log to console by adding a `StreamHandler`."""

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s %(levelname)s %(name)s - %(message)s'
            )
        )

        logger.addHandler(handler)