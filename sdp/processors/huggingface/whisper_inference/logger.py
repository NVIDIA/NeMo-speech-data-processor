import os
import logging

from sdp.logging import logger

logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
f"[%(asctime)s Rank {os.getenv('RANK', 0)}] %(levelname)s: %(message)s",
datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.handlers
logger.addHandler(handler)
logger.propagate = False