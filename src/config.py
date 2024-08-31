import os
from typing import List

from dotenv import load_dotenv

load_dotenv(".env")


class Config(object):
    # Data
    DATA_DIR: str = os.environ.get("DATA_DIR")
    DATA_FIT: str = os.environ.get("DATA_FIT")
    DATA_TEST: str = os.environ.get("DATA_TEST")
    FEATURE_COLUMNS: List[str] = os.environ.get("FEATURE_COLUMNS").split(",")
    INDEX_COLUMN: str = os.environ.get("INDEX_COLUMN")

    # Model parameters
    BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE"))
    RANDOM_STATE: int = int(os.environ.get("RANDOM_STATE"))
